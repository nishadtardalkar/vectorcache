#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/gpu/flat_index.hpp"

namespace fs = std::filesystem;

namespace {

struct Matrix {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<float> data;
};

Matrix load_npy(const fs::path& path, std::size_t limit) {
  auto reader = vectorcache::datasets::NpyReader::open(path, "npy");
  const auto meta = reader.meta();
  const std::size_t n = limit == 0 ? meta.count : std::min(limit, meta.count);
  Matrix m;
  m.rows = n;
  m.cols = meta.dim;
  m.data.resize(n * meta.dim);
  const std::size_t got = reader.copy_rows_into(m.data, n);
  m.rows = got;
  m.data.resize(got * meta.dim);
  return m;
}

Matrix load_dataset(const std::string& dataset, const fs::path& data_dir, std::size_t limit) {
  auto kind = vectorcache::datasets::parse_dataset_kind(dataset);
  if (!kind) throw std::runtime_error("unknown dataset: " + dataset);
  if (*kind == vectorcache::datasets::DatasetKind::OpenAi1536 ||
      *kind == vectorcache::datasets::DatasetKind::OpenAi3072) {
    return load_npy(vectorcache::datasets::dataset_path(*kind, data_dir), limit);
  }
  auto reader = vectorcache::datasets::OpenDatasetReader::open(
      *kind, data_dir, vectorcache::datasets::DatasetSplit::Train);
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

void l2_normalize_rows(Matrix& m) {
  for (std::size_t i = 0; i < m.rows; ++i) {
    float* row = m.data.data() + i * m.cols;
    double s = 0.0;
    for (std::size_t d = 0; d < m.cols; ++d) s += static_cast<double>(row[d]) * row[d];
    const float inv = (s > 0.0) ? static_cast<float>(1.0 / std::sqrt(s)) : 0.f;
    for (std::size_t d = 0; d < m.cols; ++d) row[d] *= inv;
  }
}

float recall1_at_k(const std::vector<std::uint64_t>& approx, const std::vector<std::size_t>& exact,
                   std::size_t k) {
  if (exact.empty() || approx.empty() || k == 0) return 0.f;
  const std::size_t e0 = exact[0];
  const std::size_t kk = std::min(k, approx.size());
  for (std::size_t i = 0; i < kk; ++i) {
    if (approx[i] == e0) return 1.f;
  }
  return 0.f;
}

std::vector<std::size_t> exact_topk(const Matrix& db, const float* q, std::size_t k) {
  std::vector<std::pair<float, std::size_t>> scored;
  scored.reserve(db.rows);
  for (std::size_t i = 0; i < db.rows; ++i) {
    const float* row = db.data.data() + i * db.cols;
    float s = 0.f;
    for (std::size_t d = 0; d < db.cols; ++d) s += row[d] * q[d];
    scored.emplace_back(s, i);
  }
  const std::size_t kk = std::min(k, scored.size());
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(kk), scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });
  std::vector<std::size_t> ids(kk);
  for (std::size_t i = 0; i < kk; ++i) ids[i] = scored[i].second;
  return ids;
}

}  // namespace

int main(int argc, char** argv) {
  std::string dataset = "openai-1536";
  fs::path data_dir = "data";
  fs::path npy;
  std::size_t limit = 100000;
  std::size_t query_limit = 1000;
  std::size_t bits = 4;
  std::size_t k = 64;
  int warmup = 2;
  int runs = 5;
  bool do_recall = false;

  CLI::App app{"vectorcache GPU full-index TurboQuant FastScan bench"};
  app.add_option("--dataset", dataset, "Dataset name");
  app.add_option("--npy", npy, "Explicit .npy path");
  app.add_option("--data-dir", data_dir, "Dataset root");
  app.add_option("--limit", limit, "Corpus rows (0=all)");
  app.add_option("--query-limit", query_limit, "Query rows");
  app.add_option("--bits", bits, "Quant bits 2-4")->check(CLI::Range(2, 4));
  app.add_option("--k", k, "Top-k");
  app.add_option("--warmup", warmup, "Warmup searches");
  app.add_option("--runs", runs, "Timed search repeats");
  app.add_flag("--recall", do_recall, "Compute 1-recall@k vs exact IP");
  CLI11_PARSE(app, argc, argv);

  if (!vectorcache::gpu::cuda_available()) {
    std::cerr << "No CUDA device available.\n";
    return 2;
  }

  Matrix db = npy.empty() ? load_dataset(dataset, data_dir, limit) : load_npy(npy, limit);
  if (db.rows == 0) {
    std::cerr << "Empty corpus.\n";
    return 1;
  }
  l2_normalize_rows(db);

  const std::size_t nq = std::min(query_limit, db.rows);
  Matrix queries;
  queries.rows = nq;
  queries.cols = db.cols;
  queries.data.assign(db.data.begin(),
                      db.data.begin() + static_cast<std::ptrdiff_t>(nq * db.cols));

  std::cout << "gpu-bench device corpus=" << db.rows << " dim=" << db.cols << " bits=" << bits
            << " nq=" << nq << " k=" << k << "\n";

  vectorcache::gpu::FlatIndex index(db.cols, bits);
  std::cout << "device: " << index.device_name() << "\n";

  const auto t0 = std::chrono::steady_clock::now();
  index.add(db.data);
  index.prepare();
  const auto t1 = std::chrono::steady_clock::now();
  const double encode_ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::cout << "encode+upload_ms=" << encode_ms << "\n";

  for (int i = 0; i < warmup; ++i) {
    (void)index.search(queries.data, k);
  }

  double total_ms = 0.0;
  vectorcache::SearchResults last;
  for (int i = 0; i < runs; ++i) {
    const auto a = std::chrono::steady_clock::now();
    last = index.search(queries.data, k);
    const auto b = std::chrono::steady_clock::now();
    total_ms += std::chrono::duration<double, std::milli>(b - a).count();
  }
  const double ms_per_query = (total_ms / runs) / static_cast<double>(nq);
  std::cout << "search_ms_total_avg=" << (total_ms / runs) << " ms_per_query=" << ms_per_query
            << "\n";

  if (do_recall) {
    double r1 = 0.0;
    const std::size_t eval_n = std::min<std::size_t>(nq, 100);
    for (std::size_t qi = 0; qi < eval_n; ++qi) {
      auto exact = exact_topk(db, queries.data.data() + qi * db.cols, k);
      std::vector<std::uint64_t> approx(k);
      for (std::size_t j = 0; j < k; ++j) approx[j] = last.ids[qi * k + j];
      r1 += recall1_at_k(approx, exact, k);
    }
    std::cout << "recall1_at_" << k << "=" << (r1 / static_cast<double>(eval_n)) << " (nq_eval="
              << eval_n << ")\n";
  }

  std::cout << "backend=gpu_fwht_lut_fastscan\n";
  return 0;
}
