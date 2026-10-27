#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

std::string cache_source_label(const fs::path& npy_path, const std::string& dataset) {
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

fs::path exact_topk_cache_path(const ExactTopkCacheKey& key) {
  const std::string name = sanitize_filename_token(key.source_label) + '_' +
                           sanitize_filename_token(key.split) + "_n" +
                           std::to_string(key.index_n) + "_qs-" +
                           sanitize_filename_token(key.query_split) + "_ql" +
                           std::to_string(key.query_n) + "_k" + std::to_string(key.k) + "_seed" +
                           std::to_string(key.seed) + ".bin";
  return fs::path(".cache") / "exact_topk" / name;
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

bool try_load_exact_topk(const fs::path& path, const ExactTopkCacheKey& key,
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

  out.assign(static_cast<std::size_t>(n_queries),
             std::vector<std::size_t>(static_cast<std::size_t>(k)));
  for (std::size_t qi = 0; qi < out.size(); ++qi) {
    for (std::size_t j = 0; j < out[qi].size(); ++j) {
      std::uint64_t id = 0;
      if (!read_u64(in, id)) {
        return false;
      }
      out[qi][j] = static_cast<std::size_t>(id);
    }
  }
  return true;
}

bool save_exact_topk(const fs::path& path, const ExactTopkCacheKey& key,
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
  fs::create_directories(path.parent_path(), ec);
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

  fs::rename(tmp, path, ec);
  if (ec) {
    fs::remove(path, ec);
    ec.clear();
    fs::rename(tmp, path, ec);
  }
  if (ec) {
    fs::remove(tmp, ec);
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Query engine benchmark (FWHT→4bit→landmarks→exact re-rank)"};

  std::string dataset = "glove";
  std::string npy;
  std::string data_dir = "data";
  std::string split = "train";
  std::string query_split;  // default: test for glove/sift1m, holdout for OpenAI / --npy
  std::size_t limit = 0;  // 0 = all
  std::size_t query_limit = 1000;
  std::size_t bits = 4;
  std::size_t rerank_bits = 0;  // 0 → same as --bits
  std::size_t k = 10;
  std::size_t landmarks = vectorcache::kDefaultLandmarks;
  bool recall = false;
  std::size_t timing_runs = 5;

  app.add_option("--dataset", dataset, "Named dataset (glove, sift1m, openai-1536, openai-3072)");
  app.add_option("--npy", npy, "Path to .npy matrix (overrides --dataset)");
  app.add_option("--data-dir", data_dir, "Dataset directory");
  app.add_option("--split", split, "Index split (train/test; HDF5 only)");
  app.add_option("--query-split", query_split,
                 "Query split: test (GloVe/SIFT1M) or holdout (OpenAI/NPY); default by dataset");
  app.add_option("--limit", limit, "Max index vectors (0 = all)");
  app.add_option("--query-limit", query_limit, "Max queries (0 = all)");
  app.add_option("--bits,--scan-bits", bits, "Bits for landmark FastScan (2-4)")
      ->check(CLI::Range(2, 4));
  app.add_option("--rerank-bits", rerank_bits,
                 "Bits for full-dim quantized re-rank (2-4; default = --bits)")
      ->check(CLI::Range(0, 4));
  app.add_option("--k", k, "Final top-k (heap size)");
  app.add_option("--landmarks", landmarks, "Orthonormal landmark subspace dims");
  app.add_flag("--recall", recall, "Compute Recall@1@k and Recall@k");
  app.add_option("--timing-runs", timing_runs, "Timed search runs (median reported)");

  CLI11_PARSE(app, argc, argv);

  try {
    auto to_split = [](const std::string& s) {
      return s == "test" ? vectorcache::datasets::DatasetSplit::Test
                         : vectorcache::datasets::DatasetSplit::Train;
    };

    auto holdout_queries = [](Matrix& db, std::size_t qlimit) -> Matrix {
      Matrix queries;
      if (qlimit > 0 && db.rows > qlimit) {
        queries.cols = db.cols;
        queries.rows = qlimit;
        queries.data.assign(db.data.end() - static_cast<std::ptrdiff_t>(qlimit * db.cols),
                            db.data.end());
        db.rows -= qlimit;
        db.data.resize(db.rows * db.cols);
      } else {
        queries = db;
        if (qlimit > 0 && queries.rows > qlimit) {
          queries.rows = qlimit;
          queries.data.resize(qlimit * queries.cols);
        }
      }
      return queries;
    };

    if (query_split.empty()) {
      const bool hdf5_test =
          npy.empty() && (dataset == "glove" || dataset == "sift1m" || dataset == "sift");
      query_split = hdf5_test ? "test" : "holdout";
    }

    Matrix db;
    Matrix queries;
    if (!npy.empty()) {
      if (query_split != "holdout") {
        throw std::runtime_error("--npy only supports --query-split holdout");
      }
      const std::size_t load_n =
          (limit == 0 || query_limit == 0) ? limit : limit + query_limit;
      db = load_npy(npy, load_n);
      queries = holdout_queries(db, query_limit);
    } else if (query_split == "holdout") {
      const std::size_t load_n =
          (limit == 0 || query_limit == 0) ? limit : limit + query_limit;
      db = load_dataset(dataset, data_dir, to_split(split), load_n);
      queries = holdout_queries(db, query_limit);
    } else if (query_split == "test" || query_split == "train") {
      db = load_dataset(dataset, data_dir, to_split(split), limit);
      queries = load_dataset(dataset, data_dir, to_split(query_split), query_limit);
    } else {
      throw std::runtime_error("unknown --query-split '" + query_split +
                               "'; use test, train, or holdout");
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

    if (landmarks == 0 || landmarks > db.cols) {
      throw std::runtime_error("--landmarks must be in 1..dim");
    }
    if (rerank_bits != 0 && (rerank_bits < 2 || rerank_bits > 4)) {
      throw std::runtime_error("--rerank-bits must be 0 (default) or 2-4");
    }
    const std::size_t effective_rerank_bits = rerank_bits == 0 ? bits : rerank_bits;

    const std::size_t n_byte_groups = landmarks / (8 / bits);
    const int planes_on =
        vectorcache::planes_wanted(bits, n_byte_groups, db.rows) ? 1 : 0;
    std::cout << "dataset rows=" << db.rows << " dim=" << db.cols << " queries=" << queries.rows
              << " scan_bits=" << bits << " rerank_bits=" << effective_rerank_bits
              << " landmarks=" << landmarks << " k=" << k << " collect=top-k-heap"
              << " search_backend=" << vectorcache::search_backend_name(bits, landmarks)
              << " planes4=" << planes_on << "\n";
    std::cout << vectorcache::search_backend_diagnostics(bits, landmarks);

    vectorcache::SearchResults last;
    std::vector<double> run_ms;
    std::vector<double> fastscan_ms;
    std::vector<double> sign_ms;
    std::vector<double> sign_vnni_ms;
    std::vector<double> sign_epilogue_ms;
    std::vector<double> exact_ms;
    std::vector<double> exact_gather_ms;
    std::vector<double> exact_alu_ms;
    std::vector<double> exact_epi_ms;

    vectorcache::LandmarkIndex index(db.cols, bits, landmarks, /*landmark_seed=*/0,
                                     effective_rerank_bits);
    const auto t0 = std::chrono::steady_clock::now();
    index.add(db.data);
    index.prepare();
    const auto t1 = std::chrono::steady_clock::now();
    std::cout << "ingest_ms=" << std::chrono::duration<double, std::milli>(t1 - t0).count() << "\n"
              << std::flush;

    if (!recall) {
      db = Matrix{};
    }

    for (std::size_t r = 0; r < timing_runs; ++r) {
      vectorcache::SearchStageTimings stages;
      const auto a = std::chrono::steady_clock::now();
      last = index.search(queries.data, k, &stages);
      const auto b = std::chrono::steady_clock::now();
      run_ms.push_back(std::chrono::duration<double, std::milli>(b - a).count());
      fastscan_ms.push_back(stages.fastscan_ms);
      sign_ms.push_back(stages.sign_ms);
      sign_vnni_ms.push_back(stages.sign_vnni_ms);
      sign_epilogue_ms.push_back(stages.sign_epilogue_ms);
      exact_ms.push_back(stages.exact_ms);
      exact_gather_ms.push_back(stages.exact_gather_ms);
      exact_alu_ms.push_back(stages.exact_alu_ms);
      exact_epi_ms.push_back(stages.exact_epi_ms);
    }

    std::sort(run_ms.begin(), run_ms.end());
    std::sort(fastscan_ms.begin(), fastscan_ms.end());
    std::sort(sign_ms.begin(), sign_ms.end());
    std::sort(sign_vnni_ms.begin(), sign_vnni_ms.end());
    std::sort(sign_epilogue_ms.begin(), sign_epilogue_ms.end());
    std::sort(exact_ms.begin(), exact_ms.end());
    std::sort(exact_gather_ms.begin(), exact_gather_ms.end());
    std::sort(exact_alu_ms.begin(), exact_alu_ms.end());
    std::sort(exact_epi_ms.begin(), exact_epi_ms.end());
    const double median_ms = run_ms[run_ms.size() / 2];
    const double median_fastscan_ms = fastscan_ms[fastscan_ms.size() / 2];
    const double median_sign_ms = sign_ms[sign_ms.size() / 2];
    const double median_sign_vnni_ms = sign_vnni_ms[sign_vnni_ms.size() / 2];
    const double median_sign_epilogue_ms = sign_epilogue_ms[sign_epilogue_ms.size() / 2];
    const double median_exact_ms = exact_ms[exact_ms.size() / 2];
    const double median_exact_gather_ms = exact_gather_ms[exact_gather_ms.size() / 2];
    const double median_exact_alu_ms = exact_alu_ms[exact_alu_ms.size() / 2];
    const double median_exact_epi_ms = exact_epi_ms[exact_epi_ms.size() / 2];
    const double nq = static_cast<double>(queries.rows);
    const double ms_per_query = median_ms / nq;
    const double fastscan_ms_per_query = median_fastscan_ms / nq;
    std::cout << "search_median_ms=" << median_ms << " ms_per_query=" << ms_per_query << "\n";
    // Codes scan only — comparable to turbovec FastScan kernel.
    std::cout << "fastscan_median_ms=" << median_fastscan_ms
              << " fastscan_ms_per_query=" << fastscan_ms_per_query << "\n";
    if (planes_on) {
      // Staged breakdown: 1-bit sign corpus scan vs 4-bit exact on shortlist.
      std::cout << "sign_median_ms=" << median_sign_ms
                << " sign_ms_per_query=" << (median_sign_ms / nq) << "\n";
      const double kernel_cpu = median_sign_vnni_ms + median_sign_epilogue_ms;
      const double vnni_frac = kernel_cpu > 0.0 ? (median_sign_vnni_ms / kernel_cpu) : 0.0;
      const double epi_frac = kernel_cpu > 0.0 ? (median_sign_epilogue_ms / kernel_cpu) : 0.0;
      // Summed over OpenMP threads (CPU ms); fractions compare VNNI vs epilogue.
      std::cout << "sign_vnni_cpu_ms=" << median_sign_vnni_ms
                << " sign_epilogue_cpu_ms=" << median_sign_epilogue_ms
                << " sign_vnni_frac=" << vnni_frac << " sign_epilogue_frac=" << epi_frac << "\n";
      std::cout << "exact_median_ms=" << median_exact_ms
                << " exact_ms_per_query=" << (median_exact_ms / nq) << "\n";
      const double exact_kernel_cpu =
          median_exact_gather_ms + median_exact_alu_ms + median_exact_epi_ms;
      std::cout << "exact_gather_cpu_ms=" << median_exact_gather_ms
                << " exact_alu_cpu_ms=" << median_exact_alu_ms
                << " exact_epi_cpu_ms=" << median_exact_epi_ms;
      if (exact_kernel_cpu > 0.0) {
        std::cout << " exact_alu_frac=" << (median_exact_alu_ms / exact_kernel_cpu)
                  << " exact_epi_frac=" << (median_exact_epi_ms / exact_kernel_cpu);
      }
      std::cout << "\n";
    }

    if (recall) {
      ExactTopkCacheKey cache_key;
      cache_key.source_label = cache_source_label(npy.empty() ? fs::path{} : fs::path(npy), dataset);
      cache_key.split = split;
      cache_key.query_split = query_split;
      cache_key.index_n = db.rows;
      cache_key.query_n = queries.rows;
      cache_key.k = k;
      cache_key.seed = 0;
      const auto cache_path = exact_topk_cache_path(cache_key);

      std::vector<std::vector<std::size_t>> exact_ids_per_query;
      if (try_load_exact_topk(cache_path, cache_key, exact_ids_per_query)) {
        std::cout << "Loaded exact top-k from " << cache_path.string() << '\n';
      } else {
        std::cout << "Computing exact top-" << k
                  << " for recall (may take several minutes)...\n";
        exact_ids_per_query.reserve(queries.rows);
        for (std::size_t qi = 0; qi < queries.rows; ++qi) {
          auto exact = exact_topk_ids(db, queries.data.data() + qi * queries.cols, k);
          if (exact.size() != k) {
            throw std::runtime_error("exact top-k size mismatch for recall cache");
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

      double r1 = 0.0;
      double rk = 0.0;
      for (std::size_t qi = 0; qi < queries.rows; ++qi) {
        const auto& exact = exact_ids_per_query[qi];
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
