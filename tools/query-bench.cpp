#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/index.hpp"
#include "vectorcache/pack/pack.hpp"

namespace fs = std::filesystem;

namespace {

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> data;
};

Matrix load_from_reader(vectorcache::datasets::DatasetReader& reader, std::size_t limit) {
  const auto meta = reader.meta();
  const std::size_t n = limit == 0 ? meta.count : std::min(limit, meta.count);
  Matrix m;
  m.rows = n;
  m.cols = meta.dim;
  m.data.resize(n * meta.dim);
  for (std::size_t i = 0; i < n; ++i) {
    if (!reader.next_vector_into(std::span<float>(m.data.data() + i * meta.dim, meta.dim))) {
      m.rows = i;
      m.data.resize(i * meta.dim);
      break;
    }
  }
  return m;
}

Matrix load_dataset(const std::string& dataset, const fs::path& data_dir,
                    vectorcache::datasets::DatasetSplit split, std::size_t limit) {
  auto kind = vectorcache::datasets::parse_dataset_kind(dataset);
  if (!kind) {
    throw std::runtime_error("unknown dataset: " + dataset);
  }
  auto reader = vectorcache::datasets::OpenDatasetReader::open(*kind, data_dir, split);
  return load_from_reader(reader, limit);
}

Matrix load_npy(const fs::path& path, std::size_t limit) {
  auto reader = vectorcache::datasets::NpyReader::open(path, "npy");
  return load_from_reader(reader, limit);
}

void l2_normalize_rows(Matrix& m) {
  for (std::size_t i = 0; i < m.rows; ++i) {
    float* row = m.data.data() + i * m.cols;
    double s = 0.0;
    for (std::size_t d = 0; d < m.cols; ++d) s += static_cast<double>(row[d]) * row[d];
    const float inv = (s > 0.0) ? static_cast<float>(1.0 / std::sqrt(s)) : 0.f;
    for (std::size_t d = 0; d < m.cols; ++d) row[d] *= inv;
  }
}

std::vector<std::size_t> exact_topk_ids(const Matrix& db, const float* query, std::size_t k) {
  std::vector<std::pair<float, std::size_t>> scored;
  scored.reserve(db.rows);
  for (std::size_t i = 0; i < db.rows; ++i) {
    const float* row = db.data.data() + i * db.cols;
    float s = 0.f;
    for (std::size_t d = 0; d < db.cols; ++d) s += row[d] * query[d];
    scored.emplace_back(s, i);
  }
  const std::size_t kk = std::min(k, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kk), scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::size_t> ids(kk);
  for (std::size_t i = 0; i < kk; ++i) ids[i] = scored[i].second;
  return ids;
}

bool recall_at_1_at_k(const std::vector<std::uint64_t>& approx, std::size_t exact_top1) {
  for (auto id : approx) {
    if (id == exact_top1) return true;
  }
  return false;
}

float recall_at_k(const std::vector<std::uint64_t>& approx, const std::vector<std::size_t>& exact) {
  std::size_t hit = 0;
  for (auto e : exact) {
    for (auto a : approx) {
      if (a == e) {
        ++hit;
        break;
      }
    }
  }
  return exact.empty() ? 0.f : static_cast<float>(hit) / static_cast<float>(exact.size());
}

fs::path exact_cache_path(const fs::path& cache_dir, const std::string& key) {
  return cache_dir / (key + ".bin");
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Query engine benchmark (TurboQuant flat index)"};

  std::string dataset = "glove";
  std::string npy;
  std::string data_dir = "data";
  std::string split = "train";
  std::string query_split = "test";
  std::size_t limit = 100000;
  std::size_t query_limit = 1000;
  std::size_t bits = 4;
  std::size_t k = 10;
  bool calibrate = false;
  bool recall = false;
  std::size_t timing_runs = 5;

  app.add_option("--dataset", dataset, "Named dataset (glove, openai-1536, openai-3072)");
  app.add_option("--npy", npy, "Path to .npy matrix (overrides --dataset)");
  app.add_option("--data-dir", data_dir, "Dataset directory");
  app.add_option("--split", split, "Index split (train/test)");
  app.add_option("--query-split", query_split, "Query split");
  app.add_option("--limit", limit, "Max index vectors (0 = all)");
  app.add_option("--query-limit", query_limit, "Max queries (0 = all)");
  app.add_option("--bits", bits, "Bits per dim (2-4)")->check(CLI::Range(2, 4));
  app.add_option("--k", k, "Top-k");
  app.add_flag("--calibrate", calibrate, "Fit TQ+ on first min(1000, n) index vectors before add");
  app.add_flag("--recall", recall, "Compute Recall@1@k and Recall@k");
  app.add_option("--timing-runs", timing_runs, "Timed search runs (median reported)");

  CLI11_PARSE(app, argc, argv);

  try {
    auto to_split = [](const std::string& s) {
      return s == "test" ? vectorcache::datasets::DatasetSplit::Test
                         : vectorcache::datasets::DatasetSplit::Train;
    };

    Matrix db;
    Matrix queries;
    if (!npy.empty()) {
      db = load_npy(npy, limit);
      // Use last query_limit rows as queries if single file, else reuse head.
      if (db.rows > query_limit && query_limit > 0) {
        queries.cols = db.cols;
        queries.rows = query_limit;
        queries.data.assign(db.data.end() - static_cast<std::ptrdiff_t>(query_limit * db.cols),
                            db.data.end());
        db.rows -= query_limit;
        db.data.resize(db.rows * db.cols);
      } else {
        queries = db;
        if (query_limit > 0 && queries.rows > query_limit) {
          queries.rows = query_limit;
          queries.data.resize(query_limit * queries.cols);
        }
      }
    } else {
      db = load_dataset(dataset, data_dir, to_split(split), limit);
      queries = load_dataset(dataset, data_dir, to_split(query_split), query_limit);
    }

    if (db.rows == 0 || queries.rows == 0) {
      std::cerr << "empty database or queries\n";
      return 1;
    }
    if (db.cols % 8 != 0) {
      std::cerr << "dim must be a multiple of 8, got " << db.cols << "\n";
      return 1;
    }

    l2_normalize_rows(db);
    l2_normalize_rows(queries);

    std::cout << "dataset rows=" << db.rows << " dim=" << db.cols << " queries=" << queries.rows
              << " bits=" << bits << " k=" << k
              << " search_backend=" << vectorcache::search_backend_name(bits, db.cols) << "\n";
    std::cout << vectorcache::search_backend_diagnostics(bits, db.cols);

    vectorcache::TurboQuantIndex index(db.cols, bits);
    if (calibrate) {
      const std::size_t n_cal =
          std::min(db.rows, static_cast<std::size_t>(vectorcache::kRecommendedCalibrationRows));
      index.calibrate(std::span<const float>(db.data.data(), n_cal * db.cols));
      std::cout << "calibrated on " << n_cal << " rows\n";
    }

    const auto t0 = std::chrono::steady_clock::now();
    index.add(db.data);
    index.prepare();
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "ingest_ms="
              << std::chrono::duration<double, std::milli>(t1 - t0).count() << "\n";

    std::vector<double> run_ms;
    vectorcache::SearchResults last;
    for (std::size_t r = 0; r < timing_runs; ++r) {
      const auto a = std::chrono::steady_clock::now();
      last = index.search(queries.data, k);
      const auto b = std::chrono::steady_clock::now();
      run_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(run_ms.begin(), run_ms.end());
    const double median_ms = run_ms[run_ms.size() / 2];
    const double ms_per_query = median_ms / static_cast<double>(queries.rows);
    std::cout << "search_median_ms=" << median_ms << " ms_per_query=" << ms_per_query << "\n";

    if (recall) {
      double r1 = 0.0;
      double rk = 0.0;
      for (std::size_t qi = 0; qi < queries.rows; ++qi) {
        auto exact = exact_topk_ids(db, queries.data.data() + qi * queries.cols, k);
        std::vector<std::uint64_t> approx(last.ids.begin() + static_cast<std::ptrdiff_t>(qi * k),
                                          last.ids.begin() + static_cast<std::ptrdiff_t>((qi + 1) * k));
        if (!exact.empty() && recall_at_1_at_k(approx, exact[0])) ++r1;
        rk += recall_at_k(approx, exact);
      }
      r1 /= static_cast<double>(queries.rows);
      rk /= static_cast<double>(queries.rows);
      std::cout << "recall@1@" << k << "=" << r1 << "\n";
      std::cout << "recall@" << k << "=" << rk << "\n";
    }

    // Print a sample of first query top-3.
    std::cout << "q0 top:";
    for (std::size_t j = 0; j < std::min<std::size_t>(3, k); ++j) {
      std::cout << " id=" << last.ids[j] << " score=" << last.scores[j];
    }
    std::cout << "\n";
  } catch (const std::exception& ex) {
    std::cerr << "error: " << ex.what() << "\n";
    return 1;
  }
  return 0;
}
