#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/datasets/sample.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/query/engine.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"

namespace {

std::pair<std::unique_ptr<vectorcache::datasets::DatasetReader>, std::string> open_reader(
    const std::filesystem::path& npy_path, const std::string& dataset,
    const std::filesystem::path& data_dir, const std::string& split) {
  if (!npy_path.empty()) {
    auto reader = std::make_unique<vectorcache::datasets::NpyReader>(
        vectorcache::datasets::NpyReader::open(npy_path, "query-bench-tune"));
    return {std::move(reader), npy_path.string()};
  }

  if (dataset.empty()) {
    throw vectorcache::Error("--dataset is required (or pass --npy)");
  }

  auto kind = vectorcache::datasets::parse_dataset_kind(dataset);
  if (!kind) {
    throw vectorcache::Error("unknown dataset '" + dataset + "'");
  }

  vectorcache::datasets::DatasetSplit dataset_split = vectorcache::datasets::DatasetSplit::Train;
  if (split == "test") {
    dataset_split = vectorcache::datasets::DatasetSplit::Test;
  } else if (split != "train") {
    throw vectorcache::Error("unknown split '" + split + "'");
  }

#ifdef VECTORCACHE_BUILD_GLOVE
  auto reader = std::make_unique<vectorcache::datasets::OpenDatasetReader>(
      vectorcache::datasets::OpenDatasetReader::open(*kind, data_dir, dataset_split));
  return {std::move(reader), dataset + " (" + split + ")"};
#else
  if (*kind == vectorcache::datasets::DatasetKind::Glove) {
    throw vectorcache::Error("GloVe requires VECTORCACHE_BUILD_GLOVE");
  }
  const auto path = vectorcache::datasets::dataset_path(*kind, data_dir);
  auto reader = std::make_unique<vectorcache::datasets::NpyReader>(
      vectorcache::datasets::NpyReader::open(path, dataset.c_str()));
  return {std::move(reader), path.string()};
#endif
}

std::vector<std::vector<float>> load_query_vectors(
    const std::string& dataset, vectorcache::datasets::DatasetReader& train_reader,
    const std::filesystem::path& data_dir, std::size_t dim, std::size_t index_limit,
    std::size_t query_limit, const std::string& query_split, std::uint64_t seed) {
  std::vector<std::vector<float>> queries;
  queries.reserve(query_limit);

  if (query_split == "test") {
    auto [test_reader, _] = open_reader({}, dataset, data_dir, "test");
    vectorcache::datasets::LimitedReader limited(*test_reader, query_limit);
    std::vector<float> buf(dim);
    while (limited.next_vector_into(buf)) {
      queries.emplace_back(buf.begin(), buf.end());
    }
    return queries;
  }

  if (query_split != "holdout") {
    throw vectorcache::Error("query_split must be 'test' or 'holdout'");
  }

  const auto meta = train_reader.meta();
  if (index_limit >= meta.count) {
    throw vectorcache::Error("index_limit must be < corpus size for holdout queries");
  }

  const auto indices =
      vectorcache::datasets::sample_index_range(index_limit, meta.count, query_limit, seed);
  vectorcache::datasets::IndexSubsetReader subset(train_reader, indices);
  std::vector<float> buf(dim);
  while (subset.next_vector_into(buf)) {
    queries.emplace_back(buf.begin(), buf.end());
  }
  return queries;
}

/// In-memory row-major float matrix reader.
class MatrixReader : public vectorcache::datasets::DatasetReader {
 public:
  MatrixReader(std::vector<float> data, std::size_t dim, std::size_t count)
      : data_(std::move(data)), dim_(dim), count_(count) {}

  vectorcache::datasets::DatasetMeta meta() const override {
    return {dim_, count_, "matrix"};
  }

  bool next_vector_into(std::span<float> out) override {
    if (index_ >= count_) {
      return false;
    }
    std::copy_n(data_.data() + index_ * dim_, dim_, out.begin());
    ++index_;
    return true;
  }

  std::span<const float> data() const { return data_; }

 private:
  std::vector<float> data_;
  std::size_t dim_;
  std::size_t count_;
  std::size_t index_ = 0;
};

std::vector<float> load_matrix(vectorcache::datasets::DatasetReader& reader, std::size_t dim,
                               std::size_t count) {
  std::vector<float> data(count * dim);
  std::vector<float> buf(dim);
  for (std::size_t i = 0; i < count; ++i) {
    if (!reader.next_vector_into(buf)) {
      throw vectorcache::Error("load_matrix: unexpected EOF");
    }
    std::copy(buf.begin(), buf.end(), data.begin() + static_cast<std::ptrdiff_t>(i * dim));
  }
  return data;
}

/// Same SplitMix64 mix as IngestionEngine (bucket_seed=0 derives from rotation seed).
std::uint64_t resolve_bucket_seed(std::uint64_t bucket_seed, std::uint64_t rotation_seed) {
  if (bucket_seed != 0) {
    return bucket_seed;
  }
  std::uint64_t z = rotation_seed + 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::vector<std::size_t> parse_size_list(const std::string& s, const char* name) {
  std::vector<std::size_t> out;
  if (s.empty()) {
    throw vectorcache::Error(std::string(name) + " list must be non-empty");
  }
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    char* end = nullptr;
    const unsigned long long v = std::strtoull(item.c_str(), &end, 10);
    if (end == item.c_str() || *end != '\0') {
      throw vectorcache::Error(std::string("invalid ") + name + " value '" + item + "'");
    }
    out.push_back(static_cast<std::size_t>(v));
  }
  if (out.empty()) {
    throw vectorcache::Error(std::string(name) + " list must be non-empty");
  }
  return out;
}

std::vector<float> parse_float_list(const std::string& s, const char* name) {
  std::vector<float> out;
  if (s.empty()) {
    throw vectorcache::Error(std::string(name) + " list must be non-empty");
  }
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (item.empty()) {
      continue;
    }
    char* end = nullptr;
    const float v = std::strtof(item.c_str(), &end);
    if (end == item.c_str() || *end != '\0' || !(v > 0.0f)) {
      throw vectorcache::Error(std::string("invalid ") + name + " value '" + item + "'");
    }
    out.push_back(v);
  }
  if (out.empty()) {
    throw vectorcache::Error(std::string(name) + " list must be non-empty");
  }
  return out;
}

bool probe_radius_ok(std::size_t P) {
  try {
    vectorcache::index::validate_probe_radius(P);
    return true;
  } catch (const vectorcache::Error&) {
    return false;
  }
}

struct Trial {
  std::size_t L = 0;
  float w = 0.0f;
  std::size_t P = 0;
  double avg_topk_mean = 0.0;
  double avg_vectors_scored = 0.0;
};

bool dominates(const Trial& a, const Trial& b) {
  const bool ge_score = a.avg_topk_mean >= b.avg_topk_mean;
  const bool le_vecs = a.avg_vectors_scored <= b.avg_vectors_scored;
  const bool strict =
      (a.avg_topk_mean > b.avg_topk_mean) || (a.avg_vectors_scored < b.avg_vectors_scored);
  return ge_score && le_vecs && strict;
}

std::vector<Trial> pareto_front(std::vector<Trial> trials) {
  std::vector<Trial> front;
  front.reserve(trials.size());
  for (const auto& t : trials) {
    bool dominated = false;
    for (const auto& other : trials) {
      if (dominates(other, t)) {
        dominated = true;
        break;
      }
    }
    if (!dominated) {
      front.push_back(t);
    }
  }
  std::sort(front.begin(), front.end(), [](const Trial& a, const Trial& b) {
    if (a.avg_topk_mean != b.avg_topk_mean) {
      return a.avg_topk_mean > b.avg_topk_mean;
    }
    if (a.avg_vectors_scored != b.avg_vectors_scored) {
      return a.avg_vectors_scored < b.avg_vectors_scored;
    }
    if (a.L != b.L) {
      return a.L < b.L;
    }
    if (a.w != b.w) {
      return a.w < b.w;
    }
    return a.P < b.P;
  });
  return front;
}

std::vector<std::uint64_t> compute_cell_keys(std::span<const float> raw_row_major, std::size_t n,
                                             std::size_t dim,
                                             const vectorcache::transform::SrhtRotation& rotation,
                                             std::size_t L, float bin_width, float fold_ridge,
                                             std::uint64_t resolved_bucket_seed) {
  const std::size_t srht_dim = rotation.srht_dim();
  const float ridge =
      fold_ridge > 0.0f ? fold_ridge : vectorcache::index::default_fold_ridge(srht_dim);
  vectorcache::index::PairHash hash(L, resolved_bucket_seed, ridge);
  std::vector<std::uint64_t> keys(n);
  std::vector<float> buf(srht_dim, 0.0f);
  for (std::size_t i = 0; i < n; ++i) {
    std::fill(buf.begin(), buf.end(), 0.0f);
    std::copy_n(raw_row_major.data() + i * dim, dim, buf.begin());
    vectorcache::transform::l2_normalize_in_place(std::span<float>(buf.data(), dim));
    rotation.apply_in_place(buf);
    keys[i] = vectorcache::index::pack_bin(
        vectorcache::index::fold_to_bin(hash, buf, bin_width));
  }
  return keys;
}

Trial run_trial(vectorcache::query::QueryEngine& engine,
                const std::vector<vectorcache::query::PreparedQuery>& prepared, std::size_t k,
                std::size_t L, float w, std::size_t P) {
  vectorcache::query::QueryParams params;
  params.k = k;
  params.probe_radius = P;

  double sum_topk_mean = 0.0;
  std::size_t scored_queries = 0;
  std::uint64_t sum_candidates = 0;

  for (const auto& pq : prepared) {
    vectorcache::query::SearchStats stats;
    const auto hits = engine.search_prepared(pq, params, &stats);
    sum_candidates += stats.candidates;
    if (!hits.empty()) {
      double hit_sum = 0.0;
      for (const auto& hit : hits) {
        hit_sum += hit.score;
      }
      sum_topk_mean += hit_sum / static_cast<double>(hits.size());
      ++scored_queries;
    }
  }

  Trial t;
  t.L = L;
  t.w = w;
  t.P = P;
  t.avg_topk_mean =
      scored_queries > 0 ? (sum_topk_mean / static_cast<double>(scored_queries)) : 0.0;
  t.avg_vectors_scored =
      prepared.empty()
          ? 0.0
          : (static_cast<double>(sum_candidates) / static_cast<double>(prepared.size()));
  return t;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Pair-hash bucket grid tuner (Pareto: max avg top-k mean, min avg vectors scored)"};
  std::filesystem::path npy_path;
  std::string dataset;
  std::filesystem::path data_dir = "data";
  std::string split = "train";
  std::string query_split;
  std::optional<std::size_t> limit;
  std::size_t query_limit = 0;
  std::uint64_t seed = 42;
  std::size_t k = 10;
  std::size_t bits = 1;
  std::size_t block_dims = 1;
  std::uint64_t bucket_seed = 0;
  float fold_ridge = 0.0f;
  std::string num_pair_dirs_list = "4,8,16";
  std::string bin_widths = "0.05,0.1,0.2,0.5";
  std::string probe_radii = "0,1,2,3";

  app.add_option("--npy", npy_path, "Pre-extracted float32 NPY matrix");
  app.add_option("--dataset", dataset, "Dataset name")->envname("VECTORCACHE_DATASET");
  app.add_option("--data-dir", data_dir, "Data directory")->envname("VECTORCACHE_DATA_DIR");
  app.add_option("--split", split, "HDF5 split for GloVe");
  app.add_option("--query-split", query_split,
                 "Query split: test (GloVe) or holdout (OpenAI); default by dataset");
  app.add_option("--limit", limit, "Cap vectors ingested into index");
  app.add_option("--query-limit", query_limit, "Query count (default: 10000 GloVe, 1000 else)");
  app.add_option("--seed", seed, "SRHT / holdout seed");
  app.add_option("--k", k, "Top-k");
  app.add_option("--bits", bits, "TurboQuantMSE bits per block (1-8; per dim when --block-dims=1)");
  app.add_option("--block-dims", block_dims, "Dims per codebook block (1-16; default 1)");
  app.add_option("--bucket-seed", bucket_seed, "Pair-hash seed (0 = derive from --seed)");
  app.add_option("--fold-ridge", fold_ridge, "Pair-hash ridge δ (0 = auto 1/srht_dim)");
  app.add_option("--num-pair-dirs-list", num_pair_dirs_list, "Comma-separated L values");
  app.add_option("--bin-widths", bin_widths,
                 "Comma-separated bin widths w on uniform u=(s+1)/2 in [0,1]");
  app.add_option("--probe-radii", probe_radii, "Comma-separated probe radii P");

  CLI11_PARSE(app, argc, argv);

  try {
    if (npy_path.empty() && dataset.empty()) {
      throw vectorcache::Error("pass --dataset or --npy");
    }
    vectorcache::quantize::validate_bits_per_dim(bits);
    vectorcache::quantize::validate_block_dims(block_dims);

    const auto L_list = parse_size_list(num_pair_dirs_list, "num-pair-dirs");
    const auto w_list = parse_float_list(bin_widths, "bin-widths");
    const auto P_list = parse_size_list(probe_radii, "probe-radii");

    for (const std::size_t L : L_list) {
      if (L == 0 || L > vectorcache::index::kMaxPairDirs) {
        throw vectorcache::Error("num-pair-dirs must be in 1..kMaxPairDirs");
      }
    }

    if (query_split.empty()) {
      query_split = (dataset == "glove") ? "test" : "holdout";
    }
    if (query_limit == 0) {
      query_limit = (dataset == "glove") ? 10000 : 1000;
    }

    auto [index_reader, source_label] = open_reader(npy_path, dataset, data_dir, split);
    const auto meta = index_reader->meta();
    const std::size_t actual_index = std::min(limit.value_or(meta.count), meta.count);
    if (actual_index == 0) {
      throw vectorcache::Error("empty index");
    }

    const std::uint64_t resolved_seed = resolve_bucket_seed(bucket_seed, seed);
    const std::size_t grid_total = L_list.size() * w_list.size() * P_list.size();

    std::cout << "Pair-hash tune: index=" << source_label << " dim=" << meta.dim
              << " index_n=" << actual_index << " query_n=" << query_limit
              << " query_split=" << query_split << " bits=" << bits << " block_dims=" << block_dims
              << " k=" << k << " grid=" << grid_total << '\n';

    // Load raw index once; ingest without buckets; rebucket from post-SRHT floats.
    vectorcache::datasets::LimitedReader index_limited(*index_reader, actual_index);
    auto raw = load_matrix(index_limited, meta.dim, actual_index);
    MatrixReader matrix_reader(std::move(raw), meta.dim, actual_index);

    vectorcache::ingest::BucketParams dummy_buckets;
    dummy_buckets.num_pair_dirs = 8;
    dummy_buckets.bin_width = 0.1f;
    dummy_buckets.fold_ridge = fold_ridge;
    dummy_buckets.bucket_seed = bucket_seed;
    auto ingest_engine = vectorcache::ingest::IngestionEngine::with_rotation(
        meta.dim, seed, bits, block_dims, dummy_buckets);

    const auto report = ingest_engine.ingest(matrix_reader, false);
    if (report.vectors_ingested != actual_index) {
      throw vectorcache::Error("index ingest count mismatch");
    }
    if (ingest_engine.store().has_buckets()) {
      throw vectorcache::Error("expected unbucketed base store");
    }

    const auto& base_store = ingest_engine.store();
    const auto raw_span = matrix_reader.data();
    const vectorcache::transform::SrhtRotation rotation(meta.dim, seed);
    const float ridge = fold_ridge > 0.0f
                            ? fold_ridge
                            : vectorcache::index::default_fold_ridge(rotation.srht_dim());

    auto [train_for_queries, _] = open_reader(npy_path, dataset, data_dir, split);
    const auto queries =
        load_query_vectors(dataset, *train_for_queries, data_dir, meta.dim, actual_index,
                           query_limit, query_split, seed);

    std::vector<Trial> trials;
    trials.reserve(grid_total);
    std::size_t evaluated = 0;
    std::size_t skipped = 0;

    std::vector<std::size_t> valid_P;
    valid_P.reserve(P_list.size());
    for (const std::size_t P : P_list) {
      if (probe_radius_ok(P)) {
        valid_P.push_back(P);
      } else {
        ++skipped;
      }
    }

    for (const std::size_t L : L_list) {
      for (const float w : w_list) {
        if (valid_P.empty()) {
          continue;
        }

        auto work = base_store.clone();
        const auto keys =
            compute_cell_keys(raw_span, actual_index, meta.dim, rotation, L, w, fold_ridge,
                              resolved_seed);
        auto hash = vectorcache::index::PairHash(L, resolved_seed, ridge);
        work.finalize_buckets(keys, std::move(hash), w);

        auto query_engine = vectorcache::query::QueryEngine::with_rotation(work, meta.dim, seed);
        query_engine.prepare_index();

        std::vector<vectorcache::query::PreparedQuery> prepared;
        prepared.reserve(queries.size());
        for (const auto& q : queries) {
          prepared.push_back(query_engine.prepare(q));
        }

        for (const std::size_t P : valid_P) {
          trials.push_back(run_trial(query_engine, prepared, k, L, w, P));
          ++evaluated;
        }
      }
    }

    std::cout << "  evaluated=" << evaluated << " skipped_invalid=" << skipped << '\n';

    const auto front = pareto_front(std::move(trials));
    std::cout << "Pareto front (avg top-k mean ↑, avg vectors scored ↓):\n";
    if (front.empty()) {
      std::cout << "  (empty)\n";
    } else {
      for (const auto& t : front) {
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  L=" << t.L << " w=" << t.w << " P=" << t.P;
        std::cout << std::setprecision(4);
        std::cout << "  topk_mean=" << t.avg_topk_mean;
        std::cout << std::setprecision(1);
        std::cout << "  vectors=" << t.avg_vectors_scored << '\n';
      }
    }

    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
