#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/datasets/sample.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/query/engine.hpp"
#include "vectorcache/query/query_config.hpp"
#include "vectorcache/transform/fwht.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"

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

struct IngestResult {
  vectorcache::ingest::IngestionEngine engine;
  std::vector<vectorcache::AlignedVector<float>> index_vectors;
};

IngestResult ingest_index(vectorcache::datasets::DatasetReader& reader, std::size_t dim,
                          std::uint64_t seed, std::size_t limit) {
  IngestResult result{vectorcache::ingest::IngestionEngine::with_rotation(dim, seed), {}};
  result.engine.reserve_vectors(limit);
  result.index_vectors.reserve(limit);

  std::vector<float> buf(dim);
  std::size_t ingested = 0;
  while (ingested < limit && reader.next_vector_into(buf)) {
    vectorcache::AlignedVector<float> rotated(result.engine.store().padded_dim(), 0.0f);
    std::copy(buf.begin(), buf.end(), rotated.begin());
    const std::size_t padded = result.engine.store().padded_dim();
    if (padded > dim) {
      std::fill(rotated.begin() + static_cast<std::ptrdiff_t>(dim), rotated.end(), 0.0f);
    }
    vectorcache::transform::l2_normalize_in_place(rotated);
    if (auto* rot = result.engine.rotation()) {
      rot->apply_in_place(rotated);
    }
    result.index_vectors.push_back(std::move(rotated));

    struct OneShotReader : vectorcache::datasets::DatasetReader {
      std::span<const float> vec;
      bool done = false;
      vectorcache::datasets::DatasetMeta meta() const override {
        return {vec.size(), 1, "one"};
      }
      bool next_vector_into(std::span<float> out) override {
        if (done) {
          return false;
        }
        std::copy(vec.begin(), vec.end(), out.begin());
        done = true;
        return true;
      }
    };

    OneShotReader one;
    one.vec = std::span<const float>(buf);
    one.done = false;

    result.engine.ingest(one);
    ++ingested;
  }

  if (ingested != limit) {
    throw vectorcache::Error("index ingest count mismatch");
  }
  return result;
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

std::vector<std::size_t> brute_force_topk(
    const vectorcache::query::PreparedQuery& query,
    const std::vector<vectorcache::AlignedVector<float>>& index_vectors, std::size_t k) {
  struct Scored {
    std::size_t id;
    float score;
  };
  std::vector<Scored> all;
  all.reserve(index_vectors.size());
  for (std::size_t i = 0; i < index_vectors.size(); ++i) {
#if VECTORCACHE_QUERY_DEPTH >= 3
    const float score = vectorcache::query::dot_f32(query.rotated, index_vectors[i]);
#else
    const float score = 0.0f;
#endif
    all.push_back({i, score});
  }
  if (k > all.size()) {
    k = all.size();
  }
  std::partial_sort(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(k), all.end(),
                    [](const Scored& a, const Scored& b) { return a.score > b.score; });
  std::vector<std::size_t> ids;
  ids.reserve(k);
  for (std::size_t i = 0; i < k; ++i) {
    ids.push_back(all[i].id);
  }
  return ids;
}

float recall_at_k(const std::vector<vectorcache::query::QueryHit>& hits,
                  const std::vector<std::size_t>& ground_truth, std::size_t k_eval) {
  if (ground_truth.empty()) {
    return 0.0f;
  }
  const std::size_t kk = std::min(k_eval, hits.size());
  const std::size_t gt_k = std::min(k_eval, ground_truth.size());

  std::size_t found = 0;
  for (std::size_t i = 0; i < gt_k; ++i) {
    const std::size_t id = ground_truth[i];
    for (std::size_t j = 0; j < kk; ++j) {
      if (hits[j].id == id) {
        ++found;
        break;
      }
    }
  }
  return static_cast<float>(found) / static_cast<float>(gt_k);
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

void run_calibration(const vectorcache::query::QueryEngine& engine,
                     const std::vector<std::vector<float>>& queries,
                     const vectorcache::query::QueryParams& base_params) {
  std::cout << "\nThreshold calibration (first query, k=" << base_params.k << "):\n";
  std::cout << "  l1_block_threshold  l0_vector_threshold  hits\n";

  const auto prepared = engine.prepare(queries.front());
  for (float l1 : {-1.0f, -0.5f, 0.0f, 0.25f, 0.5f, 0.75f}) {
    for (float l0 : {-1.0f, 0.0f, 0.25f, 0.5f, 0.75f}) {
      vectorcache::query::QueryParams p = base_params;
      p.l1_block_threshold = l1;
      p.l0_vector_threshold = l0;
      const auto hits = engine.search_prepared(prepared, p);
      std::cout << "  " << std::setw(18) << l1 << std::setw(21) << l0 << std::setw(6)
                << hits.size() << '\n';
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Query engine benchmark (recall, latency, optional calibration)"};
  std::filesystem::path npy_path;
  std::string dataset;
  std::filesystem::path data_dir = "data";
  std::string index_split = "train";
  std::string query_split;
  std::size_t index_limit = 100000;
  std::size_t query_limit = 0;
  std::uint64_t seed = 42;
  std::size_t k = 10;
  float l1_threshold = 0.0f;
  float l0_threshold = 0.0f;
  std::size_t top_blocks = 0;
  std::size_t l0_dot_promote = 0;
  bool calibrate = false;

  app.add_option("--npy", npy_path, "Pre-extracted float32 NPY matrix for index");
  app.add_option("--dataset", dataset, "Dataset name")->envname("VECTORCACHE_DATASET");
  app.add_option("--data-dir", data_dir, "Data directory")->envname("VECTORCACHE_DATA_DIR");
  app.add_option("--index-split", index_split, "Index HDF5 split (train)");
  app.add_option("--query-split", query_split,
                 "Query split: test (GloVe) or holdout (OpenAI); default by dataset");
  app.add_option("--index-limit", index_limit, "Index vectors to ingest");
  app.add_option("--query-limit", query_limit, "Query count (default: 10000 GloVe, 1000 else)");
  app.add_option("--seed", seed, "SRHT / holdout seed");
  app.add_option("--k", k, "Top-k");
  app.add_option("--l1-threshold", l1_threshold, "L1 block gate threshold");
  app.add_option("--l0-threshold", l0_threshold, "L0 vector filter threshold");
  app.add_option("--top-blocks", top_blocks, "Block routing: search top N blocks by L1 (0=all)");
  app.add_option("--l0-dot-promote", l0_dot_promote,
                 "Adaptive cascade: full dot on top M L0 survivors (0=all)");
  app.add_flag("--calibrate", calibrate, "Sweep L1/L0 thresholds on first query");

  CLI11_PARSE(app, argc, argv);

  try {
    if (npy_path.empty() && dataset.empty()) {
      throw vectorcache::Error("pass --dataset or --npy");
    }

    if (query_split.empty()) {
      query_split = (dataset == "glove") ? "test" : "holdout";
    }
    if (query_limit == 0) {
      query_limit = (dataset == "glove") ? 10000 : 1000;
    }

    auto [index_reader, source_label] = open_reader(npy_path, dataset, data_dir, index_split);
    const auto meta = index_reader->meta();
    const std::size_t actual_index = std::min(index_limit, meta.count);
    if (actual_index == 0) {
      throw vectorcache::Error("empty index");
    }

    const std::size_t padded = vectorcache::transform::padded_dim(meta.dim);
    std::cout << "Query bench: index=" << source_label << " dim=" << meta.dim
              << " padded=" << padded << " index_n=" << actual_index
              << " query_n=" << query_limit << " query_split=" << query_split << '\n';

    vectorcache::datasets::LimitedReader index_limited(*index_reader, actual_index);
    IngestResult ingested = ingest_index(index_limited, meta.dim, seed, actual_index);

    auto [train_for_queries, _] = open_reader(npy_path, dataset, data_dir, index_split);
    const auto queries =
        load_query_vectors(dataset, *train_for_queries, data_dir, meta.dim, actual_index,
                           query_limit, query_split, seed);

    auto query_engine = vectorcache::query::QueryEngine::with_rotation(ingested.engine.store(),
                                                                       meta.dim, seed);

    vectorcache::query::QueryParams params;
    params.k = k;
    params.l1_block_threshold = l1_threshold;
    params.l0_vector_threshold = l0_threshold;
    params.top_blocks = top_blocks;
    params.l0_dot_promote = l0_dot_promote;

    std::vector<std::uint64_t> prep_ns;
    std::vector<std::uint64_t> search_ns;
    prep_ns.reserve(queries.size());
    search_ns.reserve(queries.size());

    double recall1 = 0.0;
    double recall5 = 0.0;
    double recall10 = 0.0;
    std::size_t evaluated = 0;

    for (const auto& q : queries) {
      const auto t0 = Clock::now();
      const auto prepared = query_engine.prepare(q);
      const auto t1 = Clock::now();
      const auto hits = query_engine.search_prepared(prepared, params);
      const auto t2 = Clock::now();

      prep_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
      search_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()));

#if VECTORCACHE_QUERY_DEPTH >= 3
      const auto gt =
          brute_force_topk(prepared, ingested.index_vectors, std::max(k, std::size_t{10}));
      recall1 += recall_at_k(hits, gt, 1);
      recall5 += recall_at_k(hits, gt, 5);
      recall10 += recall_at_k(hits, gt, 10);
      ++evaluated;
#endif
    }

    print_latency_stats(prep_ns, search_ns);

#if VECTORCACHE_QUERY_DEPTH >= 3
    if (evaluated > 0) {
      std::cout << std::fixed << std::setprecision(4);
      std::cout << "Recall (vs brute-force on " << actual_index << " index vectors):\n";
      std::cout << "  @1:  " << (recall1 / static_cast<double>(evaluated)) << '\n';
      std::cout << "  @5:  " << (recall5 / static_cast<double>(evaluated)) << '\n';
      std::cout << "  @10: " << (recall10 / static_cast<double>(evaluated)) << '\n';
    }
#else
    std::cout << "Recall skipped (VECTORCACHE_QUERY_DEPTH < 3)\n";
#endif

    if (calibrate && !queries.empty()) {
      run_calibration(query_engine, queries, params);
    }

    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
