#include <algorithm>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/ingest/timing.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/fwht.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"
#include "vectorcache/transform/srht_config.hpp"

namespace {

#if VECTORCACHE_SRHT_ROUNDS == 1
constexpr const char* kSrhtStageLabel = "SRHT (1x sign + FWHT)";
#elif VECTORCACHE_SRHT_ROUNDS == 2
constexpr const char* kSrhtStageLabel = "SRHT (2x sign + FWHT)";
#else
constexpr const char* kSrhtStageLabel = "SRHT (3x sign + FWHT)";
#endif

struct StageTotals {
  std::uint64_t read_ns = 0;
  std::uint64_t normalize_ns = 0;
  std::uint64_t srht_ns = 0;
  std::uint64_t quantize_ns = 0;
  std::uint64_t store_ns = 0;
  std::uint64_t batch_copy_ns = 0;
  std::uint64_t vectors = 0;

  std::uint64_t total_ns() const {
    return read_ns + normalize_ns + srht_ns + quantize_ns + store_ns + batch_copy_ns;
  }
};

class LimitedReader : public vectorcache::datasets::DatasetReader {
 public:
  LimitedReader(vectorcache::datasets::DatasetReader& inner, std::size_t remaining)
      : inner_(inner), remaining_(remaining) {}

  vectorcache::datasets::DatasetMeta meta() const override { return inner_.meta(); }

  bool next_vector_into(std::span<float> out) override {
    if (remaining_ == 0) return false;
    if (!inner_.next_vector_into(out)) {
      throw vectorcache::Error("reader exhausted before reaching ingest limit");
    }
    --remaining_;
    return true;
  }

 private:
  vectorcache::datasets::DatasetReader& inner_;
  std::size_t remaining_;
};

std::pair<std::unique_ptr<vectorcache::datasets::DatasetReader>, std::string> open_reader(
    const std::filesystem::path& npy_path, const std::string& dataset,
    const std::filesystem::path& data_dir, const std::string& split) {
  if (!npy_path.empty()) {
    auto reader = std::make_unique<vectorcache::datasets::NpyReader>(
        vectorcache::datasets::NpyReader::open(npy_path, "bench"));
    return {std::move(reader), npy_path.string() + " (" + reader->meta().label + ")"};
  }

  if (dataset.empty()) {
    throw vectorcache::Error("--dataset is required (or pass --npy)");
  }

  auto kind = vectorcache::datasets::parse_dataset_kind(dataset);
  if (!kind) throw vectorcache::Error("unknown dataset '" + dataset + "'");

  vectorcache::datasets::DatasetSplit dataset_split = vectorcache::datasets::DatasetSplit::Train;
  if (split == "test") {
    dataset_split = vectorcache::datasets::DatasetSplit::Test;
  } else if (split != "train") {
    throw vectorcache::Error("unknown split '" + split + "'");
  }

#ifdef VECTORCACHE_BUILD_GLOVE
  auto reader = std::make_unique<vectorcache::datasets::OpenDatasetReader>(
      vectorcache::datasets::OpenDatasetReader::open(*kind, data_dir, dataset_split));
  return {std::move(reader), dataset + " (" + data_dir.string() + ")"};
#else
  if (*kind == vectorcache::datasets::DatasetKind::OpenAi1536 ||
      *kind == vectorcache::datasets::DatasetKind::OpenAi3072) {
    const auto path = vectorcache::datasets::dataset_path(*kind, data_dir);
    auto reader = std::make_unique<vectorcache::datasets::NpyReader>(
        vectorcache::datasets::NpyReader::open(path, vectorcache::datasets::dataset_label(*kind)));
    return {std::move(reader), dataset + " (" + path.string() + ")"};
  }
  throw vectorcache::Error("dataset requires --npy or VECTORCACHE_BUILD_GLOVE");
#endif
}

StageTotals profile_stages(vectorcache::datasets::DatasetReader& reader, std::size_t dim,
                           std::size_t padded, std::uint64_t seed, std::size_t limit) {
  const vectorcache::transform::SrhtRotation rotation(dim, seed);
  const std::size_t l1_words = vectorcache::quantize::l1_words_per_vector(padded);
  const std::size_t l0_words = vectorcache::quantize::l0_words_per_vector(padded);
  const std::size_t batch_cap =
      std::min(vectorcache::ingest::INGEST_BATCH_SIZE, std::max(limit, std::size_t{1}));

  std::vector<float> read_buf(dim);
  std::vector<float> batch_inputs(batch_cap * dim);
  std::vector<std::vector<float>> rotated(batch_cap, std::vector<float>(padded));
  std::vector<std::vector<std::uint64_t>> l1(batch_cap, std::vector<std::uint64_t>(l1_words));
  std::vector<std::vector<std::uint64_t>> l0(batch_cap, std::vector<std::uint64_t>(l0_words));

  auto store =
      vectorcache::ingest::BlockStore::with_capacity(l1_words, l0_words, padded, limit);
  StageTotals totals;
  std::size_t processed = 0;

  while (processed < limit) {
    std::size_t batch_len = 0;
    while (batch_len < batch_cap && processed + batch_len < limit) {
      const auto t0 = std::chrono::steady_clock::now();
      if (!reader.next_vector_into(read_buf)) {
        throw vectorcache::Error("reader exhausted during profiling");
      }
      totals.read_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                               t0)
              .count());
      std::copy(read_buf.begin(), read_buf.end(),
                batch_inputs.begin() + static_cast<std::ptrdiff_t>(batch_len * dim));
      ++batch_len;
    }
    if (batch_len == 0) break;

    for (std::size_t i = 0; i < batch_len; ++i) {
      const float* input =
          batch_inputs.data() + static_cast<std::ptrdiff_t>(i * dim);

      std::memcpy(rotated[i].data(), input, dim * sizeof(float));
      if (padded > dim) {
        std::memset(rotated[i].data() + dim, 0, (padded - dim) * sizeof(float));
      }

      const auto t_norm = std::chrono::steady_clock::now();
      vectorcache::transform::l2_normalize_in_place(rotated[i]);
      totals.normalize_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                               t_norm)
              .count());

      const auto t_srht = std::chrono::steady_clock::now();
      rotation.apply_in_place(rotated[i]);
      totals.srht_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                               t_srht)
              .count());

      const auto t_quant = std::chrono::steady_clock::now();
      vectorcache::quantize::quantize_4d_to_1bit_into(rotated[i], l1[i]);
      vectorcache::quantize::quantize_1dim_to_1bit_into(rotated[i], l0[i]);
      totals.quantize_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                               t_quant)
              .count());

      const auto t_store = std::chrono::steady_clock::now();
      store.push_vector(l1[i], l0[i]);
      totals.store_ns += static_cast<std::uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                               t_store)
              .count());
    }

    processed += batch_len;
    totals.vectors = processed;
  }

  return totals;
}

std::uint64_t profile_engine(vectorcache::datasets::DatasetReader& reader, std::size_t dim,
                             std::uint64_t seed, std::size_t limit) {
  LimitedReader limited(reader, limit);
  auto engine = vectorcache::ingest::IngestionEngine::with_rotation(dim, seed);
  engine.reserve_vectors(limit);
  const auto start = std::chrono::steady_clock::now();
  const auto report = engine.ingest(limited);
  const auto wall_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                           start)
          .count());
  if (report.vectors_ingested != limit) {
    throw vectorcache::Error("engine ingested unexpected vector count");
  }
  return wall_ns;
}

void print_stage_report(const std::string& label, const StageTotals& stages) {
  const std::uint64_t total = std::max<std::uint64_t>(stages.total_ns(), 1);
  const double v = std::max<std::uint64_t>(stages.vectors, 1);

  std::cout << label << '\n';
  std::cout << "  vectors: " << stages.vectors << '\n';
  std::cout << "  total:   " << vectorcache::ingest::TimingSummary::format_duration(total) << " ("
            << total << " ns)\n\n";

  const std::pair<const char*, std::uint64_t> rows[] = {
      {"read (mmap -> buffer)", stages.read_ns},
      {"batch copy (engine-style to_vec)", stages.batch_copy_ns},
      {"L2 normalize", stages.normalize_ns},
      {kSrhtStageLabel, stages.srht_ns},
      {"L1+L0 quantize", stages.quantize_ns},
      {"store (push_vector)", stages.store_ns},
  };

  std::cout << "  " << std::left << std::setw(34) << "stage" << std::right << std::setw(12)
            << "total" << std::setw(10) << "per-vec" << std::setw(7) << "share\n";
  std::cout << "  " << std::string(67, '-') << '\n';
  for (const auto& [name, ns] : rows) {
    const double pct = static_cast<double>(ns) / static_cast<double>(total) * 100.0;
    std::cout << "  " << std::left << std::setw(34) << name << std::right << std::setw(12)
              << vectorcache::ingest::TimingSummary::format_duration(ns) << std::setw(10)
              << vectorcache::ingest::TimingSummary::format_duration(
                     static_cast<std::uint64_t>(static_cast<double>(ns) / v))
              << std::setw(6) << std::fixed << std::setprecision(1) << pct << "%\n";
  }
  std::cout << '\n';
}

void print_wall_report(std::uint64_t wall_ns, std::size_t vectors, const StageTotals& stages) {
  const double v = std::max(vectors, std::size_t{1});
  const std::uint64_t seq_total = stages.total_ns();

  std::cout << "Engine ingest (release)\n";
  std::cout << "  wall:    " << vectorcache::ingest::TimingSummary::format_duration(wall_ns)
            << " (" << wall_ns << " ns)\n";
  std::cout << "  per-vec: "
            << vectorcache::ingest::TimingSummary::format_duration(
                   static_cast<std::uint64_t>(static_cast<double>(wall_ns) / v))
            << " (" << (static_cast<double>(wall_ns) / v) << " ns)\n\n";

  if (seq_total > 0) {
    const double overhead_pct =
        (static_cast<double>(wall_ns) - static_cast<double>(seq_total)) /
        static_cast<double>(seq_total) * 100.0;
    std::cout << "  engine overhead vs sequential stages: " << std::fixed << std::setprecision(1)
              << overhead_pct << "%\n";
  }

  std::cout << "\nHot paths (by sequential stage share):\n";
  std::vector<std::pair<const char*, std::uint64_t>> ranked = {
      {"SRHT / FWHT", stages.srht_ns},
      {"batch input copy (to_vec per batch)", stages.batch_copy_ns},
      {"L1+L0 quantize", stages.quantize_ns},
      {"read I/O", stages.read_ns},
      {"L2 normalize", stages.normalize_ns},
      {"store", stages.store_ns},
  };
  std::sort(ranked.begin(), ranked.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  for (std::size_t i = 0; i < ranked.size(); ++i) {
    const double pct =
        static_cast<double>(ranked[i].second) /
        static_cast<double>(std::max(seq_total, std::uint64_t{1})) *
        100.0;
    std::cout << "  " << (i + 1) << ". " << ranked[i].first << " — " << std::fixed
              << std::setprecision(1) << pct << "%\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Profile ingestion stage hot paths"};
  std::filesystem::path npy_path;
  std::string dataset;
  std::filesystem::path data_dir = "data";
  std::string split = "train";
  std::optional<std::size_t> limit;
  std::uint64_t seed = 42;

  app.add_option("--npy", npy_path, "Pre-extracted float32 NPY matrix");
  app.add_option("--dataset", dataset, "Dataset name")->envname("VECTORCACHE_DATASET");
  app.add_option("--data-dir", data_dir, "Data directory")->envname("VECTORCACHE_DATA_DIR");
  app.add_option("--split", split, "HDF5 split for GloVe");
  app.add_option("--limit", limit, "Cap vectors profiled");
  app.add_option("--seed", seed, "SRHT seed");

  CLI11_PARSE(app, argc, argv);

  try {
    if (npy_path.empty() && dataset.empty()) {
      throw vectorcache::Error("pass --dataset or --npy");
    }

    auto [reader1, source_label] = open_reader(npy_path, dataset, data_dir, split);
    const auto meta = reader1->meta();
    const std::size_t actual_limit = std::min(limit.value_or(meta.count), meta.count);
    if (actual_limit == 0) {
      throw vectorcache::Error("dataset has no vectors to profile");
    }

    const std::size_t padded = vectorcache::transform::padded_dim(meta.dim);
    std::cout << "Ingest bench: " << source_label << " (dim=" << meta.dim << ", padded=" << padded
              << ", vectors=" << actual_limit
              << ", srht_rounds=" << vectorcache::transform::srht_rounds() << ")\n";
    if (limit && *limit < meta.count) {
      std::cout << "  (capped from " << meta.count << " vectors in dataset)\n";
    }
    std::cout << '\n';

    const auto stages = profile_stages(*reader1, meta.dim, padded, seed, actual_limit);
    print_stage_report("Per-stage (sequential micro-profile)", stages);

    auto [reader2, _] = open_reader(npy_path, dataset, data_dir, split);
    const auto wall_ns = profile_engine(*reader2, meta.dim, seed, actual_limit);
    print_wall_report(wall_ns, actual_limit, stages);

    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
