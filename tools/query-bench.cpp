#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_set>
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
#include "vectorcache/transform/fwht.hpp"
#include "vectorcache/transform/normalize.hpp"

namespace {

using Clock = std::chrono::steady_clock;

std::pair<std::unique_ptr<vectorcache::datasets::DatasetReader>, std::string> open_reader(
    const std::filesystem::path& npy_path, const std::string& dataset,
    const std::filesystem::path& data_dir, const std::string& split) {
  if (!npy_path.empty()) {
    auto reader = std::make_unique<vectorcache::datasets::NpyReader>(
        vectorcache::datasets::NpyReader::open(npy_path, "query-bench"));
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

vectorcache::ingest::IngestionEngine ingest_index(vectorcache::datasets::DatasetReader& reader,
                                                  std::size_t dim, std::uint64_t seed,
                                                  std::size_t limit, std::size_t bits,
                                                  std::size_t block_dims, std::size_t num_pair_dirs,
                                                  float bin_width, std::uint64_t bucket_seed) {
  vectorcache::datasets::LimitedReader limited(reader, limit);
  vectorcache::ingest::BucketParams buckets;
  buckets.num_pair_dirs = num_pair_dirs;
  buckets.bin_width = bin_width;
  buckets.bucket_seed = bucket_seed;
  auto engine =
      vectorcache::ingest::IngestionEngine::with_rotation(dim, seed, bits, block_dims, buckets);
  engine.reserve_vectors(limit);
  const auto report = engine.ingest(limited);
  if (report.vectors_ingested != limit) {
    throw vectorcache::Error("index ingest count mismatch");
  }
  return engine;
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

/// Contiguous row-major corpus at original meta.dim; each row L2-normalized.
std::vector<float> load_normalized_corpus(vectorcache::datasets::DatasetReader& reader,
                                          std::size_t dim, std::size_t count) {
  std::vector<float> corpus(count * dim);
  vectorcache::datasets::LimitedReader limited(reader, count);
  std::size_t n = 0;
  while (n < count) {
    std::span<float> row(corpus.data() + n * dim, dim);
    if (!limited.next_vector_into(row)) {
      break;
    }
    vectorcache::transform::l2_normalize_in_place(row);
    ++n;
  }
  if (n != count) {
    throw vectorcache::Error("corpus load count mismatch");
  }
  return corpus;
}

float dot_product(std::span<const float> a, std::span<const float> b) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

/// Exact cosine top-k over L2-normalized corpus rows (original dim only).
std::vector<std::size_t> exact_topk(std::span<const float> query_norm, const std::vector<float>& corpus,
                                    std::size_t dim, std::size_t n, std::size_t k) {
  if (k == 0 || n == 0) {
    return {};
  }
  const std::size_t top_n = std::min(k, n);

  // Min-score at front when full; on equal score prefer smaller id (worse = larger id).
  struct Hit {
    float score;
    std::size_t id;
  };
  auto worse = [](const Hit& a, const Hit& b) {
    if (a.score != b.score) {
      return a.score > b.score;  // higher score is better → max-heap comparator for min-heap front
    }
    return a.id < b.id;  // smaller id is better → larger id is worse
  };

  std::vector<Hit> heap;
  heap.reserve(top_n);

  for (std::size_t id = 0; id < n; ++id) {
    const std::span<const float> row(corpus.data() + id * dim, dim);
    const float score = dot_product(query_norm, row);
    if (heap.size() < top_n) {
      heap.push_back({score, id});
      if (heap.size() == top_n) {
        std::make_heap(heap.begin(), heap.end(), worse);
      }
      continue;
    }
    const Hit& worst = heap.front();
    if (score > worst.score || (score == worst.score && id < worst.id)) {
      std::pop_heap(heap.begin(), heap.end(), worse);
      heap.back() = {score, id};
      std::push_heap(heap.begin(), heap.end(), worse);
    }
  }

  std::sort_heap(heap.begin(), heap.end(), worse);
  // `worse` treats higher score as "less", so sort_heap places the best hit first.

  std::vector<std::size_t> ids;
  ids.reserve(heap.size());
  for (const auto& h : heap) {
    ids.push_back(h.id);
  }
  return ids;
}

double recall_at_k(const std::vector<std::size_t>& approx_ids,
                   const std::vector<std::size_t>& exact_ids, std::size_t k) {
  if (k == 0) {
    return 0.0;
  }
  std::unordered_set<std::size_t> exact_set(exact_ids.begin(), exact_ids.end());
  std::size_t hits = 0;
  for (const std::size_t id : approx_ids) {
    if (exact_set.contains(id)) {
      ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(k);
}

/// TurboVec-style recall@1@k: exact NN id present in approx top-k.
bool recall_at_1_at_k(const std::vector<std::size_t>& approx_ids, std::size_t exact_top1) {
  return std::find(approx_ids.begin(), approx_ids.end(), exact_top1) != approx_ids.end();
}

struct ExactTopkCacheKey {
  std::string source_label;
  std::string split;
  std::string query_split;
  std::uint64_t index_n = 0;
  std::uint64_t query_n = 0;
  std::uint64_t k = 0;
  std::uint64_t seed = 0;
};

std::string sanitize_filename_token(std::string s) {
  for (char& c : s) {
    if (!std::isalnum(static_cast<unsigned char>(c)) && c != '-' && c != '_') {
      c = '_';
    }
  }
  return s;
}

std::string cache_source_label(const std::filesystem::path& npy_path, const std::string& dataset) {
  if (npy_path.empty()) {
    return dataset;
  }
  std::uint64_t h = 14695981039346656037ULL;
  const auto path_str = npy_path.lexically_normal().string();
  for (unsigned char c : path_str) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  std::ostringstream oss;
  oss << "npy-" << npy_path.stem().string() << '-' << std::hex << h;
  return oss.str();
}

std::filesystem::path exact_topk_cache_path(const ExactTopkCacheKey& key) {
  const std::string name = sanitize_filename_token(key.source_label) + '_' +
                           sanitize_filename_token(key.split) + "_n" +
                           std::to_string(key.index_n) + "_qs-" +
                           sanitize_filename_token(key.query_split) + "_ql" +
                           std::to_string(key.query_n) + "_k" + std::to_string(key.k) + "_seed" +
                           std::to_string(key.seed) + ".bin";
  return std::filesystem::path(".cache") / "exact_topk" / name;
}

bool write_u32(std::ostream& out, std::uint32_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
  return static_cast<bool>(out);
}

bool write_u64(std::ostream& out, std::uint64_t v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(v));
  return static_cast<bool>(out);
}

bool write_string(std::ostream& out, const std::string& s) {
  if (s.size() > 0xffffffffu) {
    return false;
  }
  if (!write_u32(out, static_cast<std::uint32_t>(s.size()))) {
    return false;
  }
  out.write(s.data(), static_cast<std::streamsize>(s.size()));
  return static_cast<bool>(out);
}

bool read_u32(std::istream& in, std::uint32_t& v) {
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return static_cast<bool>(in);
}

bool read_u64(std::istream& in, std::uint64_t& v) {
  in.read(reinterpret_cast<char*>(&v), sizeof(v));
  return static_cast<bool>(in);
}

bool read_string(std::istream& in, std::string& s) {
  std::uint32_t len = 0;
  if (!read_u32(in, len)) {
    return false;
  }
  s.resize(len);
  if (len == 0) {
    return true;
  }
  in.read(s.data(), static_cast<std::streamsize>(len));
  return static_cast<bool>(in);
}

bool try_load_exact_topk(const std::filesystem::path& path, const ExactTopkCacheKey& key,
                         std::vector<std::vector<std::size_t>>& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return false;
  }

  char magic[4] = {};
  in.read(magic, 4);
  if (!in || std::memcmp(magic, "VCEX", 4) != 0) {
    return false;
  }
  std::uint32_t version = 0;
  if (!read_u32(in, version) || version != 1) {
    return false;
  }

  std::uint64_t n_queries = 0;
  std::uint64_t k = 0;
  std::uint64_t index_n = 0;
  std::uint64_t seed = 0;
  if (!read_u64(in, n_queries) || !read_u64(in, k) || !read_u64(in, index_n) ||
      !read_u64(in, seed)) {
    return false;
  }
  if (n_queries != key.query_n || k != key.k || index_n != key.index_n || seed != key.seed) {
    return false;
  }

  std::string source_label;
  std::string split;
  std::string query_split;
  if (!read_string(in, source_label) || !read_string(in, split) || !read_string(in, query_split)) {
    return false;
  }
  if (source_label != key.source_label || split != key.split || query_split != key.query_split) {
    return false;
  }

  out.assign(static_cast<std::size_t>(n_queries), std::vector<std::size_t>(static_cast<std::size_t>(k)));
  for (std::size_t qi = 0; qi < out.size(); ++qi) {
    for (std::size_t j = 0; j < out[qi].size(); ++j) {
      std::uint64_t id = 0;
      if (!read_u64(in, id)) {
        return false;
      }
      out[qi][j] = static_cast<std::size_t>(id);
    }
  }
  // Trailing bytes are ignored; require we consumed at least the payload.
  return true;
}

bool save_exact_topk(const std::filesystem::path& path, const ExactTopkCacheKey& key,
                     const std::vector<std::vector<std::size_t>>& exact_ids) {
  if (exact_ids.size() != key.query_n) {
    return false;
  }
  for (const auto& row : exact_ids) {
    if (row.size() != key.k) {
      return false;
    }
  }

  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  if (ec) {
    return false;
  }

  const auto tmp = path.string() + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) {
      return false;
    }
    out.write("VCEX", 4);
    if (!write_u32(out, 1) || !write_u64(out, key.query_n) || !write_u64(out, key.k) ||
        !write_u64(out, key.index_n) || !write_u64(out, key.seed) ||
        !write_string(out, key.source_label) || !write_string(out, key.split) ||
        !write_string(out, key.query_split)) {
      return false;
    }
    for (const auto& row : exact_ids) {
      for (const std::size_t id : row) {
        if (!write_u64(out, static_cast<std::uint64_t>(id))) {
          return false;
        }
      }
    }
    if (!out.flush()) {
      return false;
    }
  }

  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    std::filesystem::remove(path, ec);
    ec.clear();
    std::filesystem::rename(tmp, path, ec);
  }
  if (ec) {
    std::filesystem::remove(tmp, ec);
    return false;
  }
  return true;
}

void print_score_stats(double sum_top1, double sum_topk_mean, std::size_t queries,
                       std::size_t k) {
  if (queries == 0) {
    return;
  }
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "Scores (similarity, k=" << k << "):\n";
  std::cout << "  avg top-1:      " << (sum_top1 / static_cast<double>(queries)) << '\n';
  std::cout << "  avg top-k mean: " << (sum_topk_mean / static_cast<double>(queries)) << '\n';
}

void print_latency_stats(const std::vector<std::uint64_t>& prep_ns,
                         const std::vector<std::uint64_t>& search_ns) {
  auto percentile = [](std::vector<std::uint64_t> v, double p) -> double {
    if (v.empty()) {
      return 0.0;
    }
    std::sort(v.begin(), v.end());
    const std::size_t idx =
        std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size() - 1)));
    return static_cast<double>(v[idx]) / 1e6;
  };

  std::vector<std::uint64_t> total;
  total.reserve(prep_ns.size());
  for (std::size_t i = 0; i < prep_ns.size(); ++i) {
    total.push_back(prep_ns[i] + search_ns[i]);
  }

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Latency ms (prep / search / total):\n";
  std::cout << "  p50: " << percentile(prep_ns, 0.50) << " / " << percentile(search_ns, 0.50)
            << " / " << percentile(total, 0.50) << '\n';
  std::cout << "  p99: " << percentile(prep_ns, 0.99) << " / " << percentile(search_ns, 0.99)
            << " / " << percentile(total, 0.99) << '\n';

  const double total_s =
      static_cast<double>(
          std::accumulate(total.begin(), total.end(), static_cast<std::uint64_t>(0))) /
      1e9;
  if (total_s > 0.0) {
    std::cout << "  QPS: " << (static_cast<double>(total.size()) / total_s) << '\n';
  }
}

void run_probe_stats(const vectorcache::query::QueryEngine& engine,
                     const std::vector<std::vector<float>>& queries,
                     const vectorcache::query::QueryParams& base_params) {
  std::cout << "\nProbe stats (first query, k=" << base_params.k << "):\n";
  const auto prepared = engine.prepare(queries.front());
  const auto hits = engine.search_prepared(prepared, base_params);
  std::cout << "  rotated_dim=" << prepared.rotated.size()
            << " bits=" << engine.codebook().bits() << " hits=" << hits.size() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Query engine benchmark (scores, latency, optional probe stats)"};
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
  std::size_t num_pair_dirs = 8;
  float bin_width = 0.1f;
  std::size_t probe_radius = 1;
  std::uint64_t bucket_seed = 0;
  bool calibrate = false;
  bool recall = false;

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
  app.add_option("--num-pair-dirs", num_pair_dirs, "Pair-hash random 2D directions L (1-64)");
  app.add_option("--bin-width", bin_width, "Pair-hash bin width w on arcsine-CDF u in [0,1]");
  app.add_option("--probe-radius", probe_radius, "1D multi-probe radius P");
  app.add_option("--bucket-seed", bucket_seed, "Pair-hash seed (0 = derive from --seed)");
  app.add_flag("--calibrate", calibrate, "Print L0 probe stats for the first query");
  app.add_flag("--recall", recall,
               "Measure Recall@1@k (exact NN in approx top-k; TurboVec-compatible) and "
               "set-overlap Recall@k vs exact cosine on original full-dim vectors");

  CLI11_PARSE(app, argc, argv);

  try {
    if (npy_path.empty() && dataset.empty()) {
      throw vectorcache::Error("pass --dataset or --npy");
    }
    vectorcache::quantize::validate_bits_per_dim(bits);
    vectorcache::quantize::validate_block_dims(block_dims);
    if (num_pair_dirs == 0 || num_pair_dirs > vectorcache::index::kMaxPairDirs) {
      throw vectorcache::Error("num-pair-dirs must be in 1..kMaxPairDirs");
    }
    if (!(bin_width > 0.0f)) {
      throw vectorcache::Error("bin-width must be > 0");
    }
    vectorcache::index::validate_probe_radius(probe_radius);

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

    std::cout << "Query bench: index=" << source_label << " dim=" << meta.dim
              << " srht_dim=" << meta.dim << " index_n=" << actual_index
              << " query_n=" << query_limit << " query_split=" << query_split
              << " bits=" << bits << " block_dims=" << block_dims
              << " L=" << num_pair_dirs << " w=" << bin_width << " P=" << probe_radius << '\n';
    if (limit && *limit < meta.count) {
      std::cout << "  (index capped from " << meta.count << " vectors in dataset)\n";
    }

    vectorcache::datasets::LimitedReader index_limited(*index_reader, actual_index);
    auto ingest_engine =
        ingest_index(index_limited, meta.dim, seed, actual_index, bits, block_dims, num_pair_dirs,
                     bin_width, bucket_seed);
    {
      const auto& store = ingest_engine.store();
      std::cout << "  stored_vectors=" << store.size()
                << " bucket_cells=" << store.buckets().num_cells() << '\n';
      const auto& buckets0 = store.buckets();
      std::vector<std::size_t> sizes;
      sizes.reserve(buckets0.num_cells());
      for (std::size_t i = 0; i < buckets0.num_cells(); ++i) {
        sizes.push_back(buckets0.cell(i).length);
      }
      std::sort(sizes.begin(), sizes.end(), std::greater<>());
      std::cout << "  bucket_sizes (desc):";
      for (const std::size_t n : sizes) {
        std::cout << ' ' << n;
      }
      std::cout << '\n';
    }

    auto [train_for_queries, _] = open_reader(npy_path, dataset, data_dir, split);
    const auto queries =
        load_query_vectors(dataset, *train_for_queries, data_dir, meta.dim, actual_index,
                           query_limit, query_split, seed);

    auto query_engine =
        vectorcache::query::QueryEngine::with_rotation(ingest_engine.store(), meta.dim, seed);

    vectorcache::query::QueryParams params;
    params.k = k;
    params.probe_radius = probe_radius;

    std::vector<std::uint64_t> prep_ns;
    std::vector<std::uint64_t> search_ns;
    prep_ns.reserve(queries.size());
    search_ns.reserve(queries.size());

    std::vector<std::vector<std::size_t>> approx_ids_per_query;
    if (recall) {
      approx_ids_per_query.reserve(queries.size());
    }

    double sum_top1 = 0.0;
    double sum_topk_mean = 0.0;
    std::size_t scored_queries = 0;
    std::uint64_t sum_candidates = 0;
    std::uint64_t sum_cells = 0;

    // Warm up + reuse PreparedQuery buffers so prep latency excludes allocation.
    vectorcache::query::PreparedQuery prepared;
    if (!queries.empty()) {
      query_engine.prepare_into(prepared, queries.front());
      vectorcache::query::SearchStats warm_stats;
      (void)query_engine.search_prepared(prepared, params, &warm_stats);
    }

    for (const auto& q : queries) {
      const auto t0 = Clock::now();
      query_engine.prepare_into(prepared, q);
      const auto t1 = Clock::now();
      vectorcache::query::SearchStats stats;
      const auto hits = query_engine.search_prepared(prepared, params, &stats);
      const auto t2 = Clock::now();

      prep_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
      search_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()));
      sum_candidates += stats.candidates;
      sum_cells += stats.cells_probed;

      if (recall) {
        std::vector<std::size_t> ids;
        ids.reserve(hits.size());
        for (const auto& hit : hits) {
          ids.push_back(hit.id);
        }
        approx_ids_per_query.push_back(std::move(ids));
      }

      if (!hits.empty()) {
        sum_top1 += hits.front().score;
        double hit_sum = 0.0;
        for (const auto& hit : hits) {
          hit_sum += hit.score;
        }
        sum_topk_mean += hit_sum / static_cast<double>(hits.size());
        ++scored_queries;
      }
    }

    print_latency_stats(prep_ns, search_ns);
    print_score_stats(sum_top1, sum_topk_mean, scored_queries, k);
    if (!queries.empty()) {
      const double avg_scored =
          static_cast<double>(sum_candidates) / static_cast<double>(queries.size());
      const double pct_scored =
          actual_index > 0 ? (100.0 * avg_scored / static_cast<double>(actual_index)) : 0.0;
      std::cout << std::fixed << std::setprecision(1);
      std::cout << "Bucket prune:\n";
      std::cout << "  avg vectors scored: " << avg_scored << " / " << actual_index << " ("
                << pct_scored << "% of index)\n";
      std::cout << "  avg cells probed: "
                << (static_cast<double>(sum_cells) / static_cast<double>(queries.size())) << '\n';
    }

    if (recall) {
      ExactTopkCacheKey cache_key;
      cache_key.source_label = cache_source_label(npy_path, dataset);
      cache_key.split = split;
      cache_key.query_split = query_split;
      cache_key.index_n = actual_index;
      cache_key.query_n = queries.size();
      cache_key.k = k;
      cache_key.seed = seed;
      const auto cache_path = exact_topk_cache_path(cache_key);

      std::vector<std::vector<std::size_t>> exact_ids_per_query;
      if (try_load_exact_topk(cache_path, cache_key, exact_ids_per_query)) {
        std::cout << "Loaded exact top-k from " << cache_path.string() << '\n';
      } else {
        std::cout << "Computing exact top-" << k
                  << " for recall (original dim; may take several minutes)...\n";
        auto [corpus_reader, corpus_label] = open_reader(npy_path, dataset, data_dir, split);
        (void)corpus_label;
        const std::vector<float> corpus =
            load_normalized_corpus(*corpus_reader, meta.dim, actual_index);

        exact_ids_per_query.reserve(queries.size());
        std::vector<float> q_norm(meta.dim);
        for (std::size_t qi = 0; qi < queries.size(); ++qi) {
          const auto& q = queries[qi];
          if (q.size() != meta.dim) {
            throw vectorcache::Error("query dimension mismatch for recall");
          }
          std::copy(q.begin(), q.end(), q_norm.begin());
          vectorcache::transform::l2_normalize_in_place(q_norm);
          auto exact = exact_topk(q_norm, corpus, meta.dim, actual_index, k);
          if (exact.empty()) {
            throw vectorcache::Error("exact top-k empty for recall");
          }
          if (exact.size() != k) {
            throw vectorcache::Error("exact top-k size mismatch for recall cache");
          }
          exact_ids_per_query.push_back(std::move(exact));
        }
        if (save_exact_topk(cache_path, cache_key, exact_ids_per_query)) {
          std::cout << "Wrote exact top-k to " << cache_path.string() << '\n';
        } else {
          std::cerr << "Warning: failed to write exact top-k cache " << cache_path.string()
                    << '\n';
        }
      }

      double sum_recall = 0.0;
      double sum_r1atk = 0.0;
      std::size_t recall_queries = 0;
      for (std::size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& exact = exact_ids_per_query[qi];
        if (exact.empty()) {
          throw vectorcache::Error("exact top-k empty for recall");
        }
        sum_r1atk += recall_at_1_at_k(approx_ids_per_query[qi], exact.front()) ? 1.0 : 0.0;
        sum_recall += recall_at_k(approx_ids_per_query[qi], exact, k);
        ++recall_queries;
      }

      if (recall_queries > 0) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "Recall@1@" << k << ": "
                  << (sum_r1atk / static_cast<double>(recall_queries)) << '\n';
        std::cout << "Recall@" << k << ": "
                  << (sum_recall / static_cast<double>(recall_queries)) << '\n';
      }
    }

    if (calibrate && !queries.empty()) {
      run_probe_stats(query_engine, queries, params);
    }

    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
