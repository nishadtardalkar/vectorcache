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
#include <unordered_set>
#include <utility>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/datasets/sample.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
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
                                                   std::size_t limit) {
  vectorcache::datasets::LimitedReader limited(reader, limit);
  auto engine = vectorcache::ingest::IngestionEngine::with_rotation(dim, seed);
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
  // After sort_heap with worse (max-oriented), ascending by worse means best last; reverse.
  std::reverse(heap.begin(), heap.end());

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
  std::cout << "  parent_key=0x" << std::hex << std::setw(2) << std::setfill('0')
            << static_cast<unsigned>(prepared.parent_key) << std::dec
            << " hits=" << hits.size() << '\n';
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
  app.add_flag("--calibrate", calibrate, "Print parent-key probe stats for the first query");
  app.add_flag("--recall", recall,
               "Measure mean recall@k vs exact cosine top-k on original full-dim vectors");

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

    auto [index_reader, source_label] = open_reader(npy_path, dataset, data_dir, split);
    const auto meta = index_reader->meta();
    const std::size_t actual_index = std::min(limit.value_or(meta.count), meta.count);
    if (actual_index == 0) {
      throw vectorcache::Error("empty index");
    }

    const std::size_t padded = vectorcache::transform::padded_dim(meta.dim);
    std::cout << "Query bench: index=" << source_label << " dim=" << meta.dim
              << " padded=" << padded << " index_n=" << actual_index
              << " query_n=" << query_limit << " query_split=" << query_split << '\n';
    if (limit && *limit < meta.count) {
      std::cout << "  (index capped from " << meta.count << " vectors in dataset)\n";
    }

    vectorcache::datasets::LimitedReader index_limited(*index_reader, actual_index);
    auto ingest_engine = ingest_index(index_limited, meta.dim, seed, actual_index);
    std::cout << "  unique_parents=" << ingest_engine.store().unique_parent_count() << '\n';

    auto [train_for_queries, _] = open_reader(npy_path, dataset, data_dir, split);
    const auto queries =
        load_query_vectors(dataset, *train_for_queries, data_dir, meta.dim, actual_index,
                           query_limit, query_split, seed);

    auto query_engine = vectorcache::query::QueryEngine::with_rotation(ingest_engine.store(),
                                                                       meta.dim, seed);

    vectorcache::query::QueryParams params;
    params.k = k;

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

    // Warm up + reuse PreparedQuery buffers so prep latency excludes allocation.
    vectorcache::query::PreparedQuery prepared;
    if (!queries.empty()) {
      query_engine.prepare_into(prepared, queries.front());
      (void)query_engine.search_prepared(prepared, params);
    }

    for (const auto& q : queries) {
      const auto t0 = Clock::now();
      query_engine.prepare_into(prepared, q);
      const auto t1 = Clock::now();
      const auto hits = query_engine.search_prepared(prepared, params);
      const auto t2 = Clock::now();

      prep_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()));
      search_ns.push_back(static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()));

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

    if (recall) {
      std::cout << "Computing exact top-" << k
                << " for recall (original dim; may take several minutes)...\n";
      auto [corpus_reader, corpus_label] = open_reader(npy_path, dataset, data_dir, split);
      (void)corpus_label;
      const std::vector<float> corpus =
          load_normalized_corpus(*corpus_reader, meta.dim, actual_index);

      double sum_recall = 0.0;
      std::size_t recall_queries = 0;
      std::vector<float> q_norm(meta.dim);
      for (std::size_t qi = 0; qi < queries.size(); ++qi) {
        const auto& q = queries[qi];
        if (q.size() != meta.dim) {
          throw vectorcache::Error("query dimension mismatch for recall");
        }
        std::copy(q.begin(), q.end(), q_norm.begin());
        vectorcache::transform::l2_normalize_in_place(q_norm);
        const auto exact =
            exact_topk(q_norm, corpus, meta.dim, actual_index, k);
        sum_recall += recall_at_k(approx_ids_per_query[qi], exact, k);
        ++recall_queries;
      }

      if (recall_queries > 0) {
        std::cout << std::fixed << std::setprecision(4);
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
