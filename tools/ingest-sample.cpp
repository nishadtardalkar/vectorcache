#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>

#include "vectorcache/datasets/datasets.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/ingest/timing.hpp"
#include "vectorcache/transform/fwht.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"

namespace {

double variance_across_dims(std::span<const float> vector) {
  if (vector.empty()) return 0.0;
  const double n = static_cast<double>(vector.size());
  double mean = 0.0;
  for (float x : vector) mean += static_cast<double>(x);
  mean /= n;
  double sum_sq = 0.0;
  for (float x : vector) {
    const double d = static_cast<double>(x) - mean;
    sum_sq += d * d;
  }
  return sum_sq / n;
}

void print_variance_stats(const std::string& label, const std::vector<double>& variances) {
  if (variances.empty()) {
    std::cout << label << ": (no vectors)\n";
    return;
  }
  double avg = 0.0;
  double min_v = std::numeric_limits<double>::infinity();
  double max_v = -std::numeric_limits<double>::infinity();
  for (double v : variances) {
    avg += v;
    min_v = std::min(min_v, v);
    max_v = std::max(max_v, v);
  }
  avg /= static_cast<double>(variances.size());
  std::cout << label << " per-vector variance across dims:\n";
  std::cout << "  average: " << std::fixed << std::setprecision(8) << avg << '\n';
  std::cout << "  min:     " << min_v << '\n';
  std::cout << "  max:     " << max_v << '\n';
}

class VarianceHook : public vectorcache::ingest::VectorHook {
 public:
  explicit VarianceHook(bool capture_vectors) : capture_vectors_(capture_vectors) {}

  void on_vector(std::uint64_t /*global_id*/, std::span<const float> vector) override {
    post_variances_.push_back(variance_across_dims(vector));
    if (capture_vectors_) {
      vectors_.emplace_back(vector.begin(), vector.end());
    }
  }

  const std::vector<double>& post_variances() const { return post_variances_; }
  const std::vector<std::vector<float>>& vectors() const { return vectors_; }

 private:
  bool capture_vectors_;
  std::vector<double> post_variances_;
  std::vector<std::vector<float>> vectors_;
};

class LimitedReader : public vectorcache::datasets::DatasetReader {
 public:
  LimitedReader(vectorcache::datasets::DatasetReader& inner, std::size_t remaining,
                std::vector<double>* pre_variances)
      : inner_(inner), remaining_(remaining), pre_variances_(pre_variances) {}

  vectorcache::datasets::DatasetMeta meta() const override { return inner_.meta(); }

  bool next_vector_into(std::span<float> out) override {
    if (remaining_ == 0) return false;
    if (!inner_.next_vector_into(out)) {
      throw vectorcache::Error("reader exhausted before reaching ingest limit");
    }
    if (pre_variances_ != nullptr) {
      pre_variances_->push_back(variance_across_dims(out));
    }
    --remaining_;
    return true;
  }

 private:
  vectorcache::datasets::DatasetReader& inner_;
  std::size_t remaining_;
  std::vector<double>* pre_variances_;
};

class MultiRoundReader : public vectorcache::datasets::DatasetReader {
 public:
  MultiRoundReader(vectorcache::datasets::DatasetReader& inner, std::size_t remaining,
                   std::vector<double>* pre_variances, std::size_t rounds, std::uint64_t seed,
                   std::vector<std::vector<double>>* round_variances)
      : inner_(inner),
        remaining_(remaining),
        pre_variances_(pre_variances),
        rounds_(rounds),
        seed_(seed),
        round_variances_(round_variances) {
    raw_.resize(inner.meta().dim);
    output_.resize(inner.meta().dim);
  }

  vectorcache::datasets::DatasetMeta meta() const override { return inner_.meta(); }

  bool next_vector_into(std::span<float> out) override {
    if (remaining_ == 0) return false;
    if (!inner_.next_vector_into(raw_)) {
      throw vectorcache::Error("reader exhausted before reaching ingest limit");
    }
    if (pre_variances_ != nullptr) {
      pre_variances_->push_back(variance_across_dims(raw_));
    }
    --remaining_;

    output_ = raw_;
    vectorcache::transform::l2_normalize_in_place(output_);
    for (std::size_t round = 0; round < rounds_; ++round) {
      vectorcache::transform::SrhtRotation rot(output_.size(), seed_ + round);
      scratch_.assign(rot.padded_dim(), 0.0f);
      rot.apply(output_, scratch_);
      if (round_variances_ != nullptr && !round_variances_->empty()) {
        (*round_variances_)[round].push_back(variance_across_dims(scratch_));
      }
      output_ = scratch_;
    }

    if (out.size() != output_.size()) {
      throw vectorcache::Error("output buffer dimension mismatch");
    }
    std::copy(output_.begin(), output_.end(), out.begin());
    return true;
  }

 private:
  vectorcache::datasets::DatasetReader& inner_;
  std::size_t remaining_;
  std::vector<double>* pre_variances_;
  std::size_t rounds_;
  std::uint64_t seed_;
  std::vector<std::vector<double>>* round_variances_;
  std::vector<float> raw_;
  std::vector<float> scratch_;
  std::vector<float> output_;
};

void print_stored_l1_codes(std::size_t index, const vectorcache::ingest::IngestionEngine& engine,
                           std::size_t padded_dim) {
  const std::size_t words_per_vec = engine.store().l1_words_per_vec();
  const std::size_t num_bits = (padded_dim + 3) / 4;
  const std::size_t block_idx = index / vectorcache::ingest::BLOCK_SIZE;
  const std::size_t vec_in_block = index % vectorcache::ingest::BLOCK_SIZE;

  const vectorcache::ingest::VectorBlock* block = nullptr;
  if (block_idx < engine.store().block_count()) {
    block = engine.store().get_block(block_idx);
  } else {
    block = &engine.store().partial_block();
  }

  const auto slice = block->as_slice();
  const std::size_t offset = vec_in_block * words_per_vec;
  std::cout << "Stored L1 codes at index " << index << " (" << num_bits << " bits, "
            << words_per_vec << " u64 words):\n  [";
  for (std::size_t i = 0; i < words_per_vec; ++i) {
    if (i > 0) std::cout << ", ";
    std::cout << "0x" << std::hex << std::setw(16) << std::setfill('0') << slice[offset + i];
  }
  std::cout << std::dec << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"Ingest vectors and optionally report per-vector dimension variance"};
  std::filesystem::path npy_path;
  std::string dataset;
  std::filesystem::path data_dir = "data";
  std::size_t limit = 100;
  std::string split = "train";
  std::uint64_t seed = 42;
  std::size_t rounds = 1;
  bool variance = false;
  std::optional<std::size_t> show_index;

  app.add_option("--npy", npy_path, "Pre-extracted float32 NPY matrix");
  app.add_option("--dataset", dataset, "Dataset name")->envname("VECTORCACHE_DATASET");
  app.add_option("--data-dir", data_dir, "Data directory")->envname("VECTORCACHE_DATA_DIR");
  app.add_option("--limit", limit, "Maximum number of vectors to ingest");
  app.add_option("--split", split, "HDF5 split for GloVe (train or test)");
  app.add_option("--seed", seed, "SRHT rotation seed");
  app.add_option("--rounds", rounds, "Number of consecutive SRHT rounds");
  app.add_flag("--variance", variance, "Report per-vector dimension variance");
  app.add_option("--show-index", show_index, "Print stored vector at index");

  CLI11_PARSE(app, argc, argv);

  try {
    if (rounds == 0) {
      throw vectorcache::Error("--rounds must be at least 1");
    }

    vectorcache::datasets::DatasetSplit dataset_split = vectorcache::datasets::DatasetSplit::Train;
    if (split == "test") {
      dataset_split = vectorcache::datasets::DatasetSplit::Test;
    } else if (split != "train") {
      throw vectorcache::Error("unknown split '" + split + "'; use train or test");
    }

    std::unique_ptr<vectorcache::datasets::DatasetReader> owned_reader;
    vectorcache::datasets::DatasetReader* reader_ptr = nullptr;

    if (!npy_path.empty()) {
      owned_reader =
          std::make_unique<vectorcache::datasets::NpyReader>(
              vectorcache::datasets::NpyReader::open(npy_path, "glove-sample"));
      reader_ptr = owned_reader.get();
    } else {
      const std::string name = dataset.empty() ? "glove" : dataset;
      auto kind = vectorcache::datasets::parse_dataset_kind(name);
      if (!kind) {
        throw vectorcache::Error("unknown dataset '" + name + "'");
      }
      owned_reader = std::make_unique<vectorcache::datasets::OpenDatasetReader>(
          vectorcache::datasets::OpenDatasetReader::open(*kind, data_dir, dataset_split));
      reader_ptr = owned_reader.get();
    }

    const auto meta = reader_ptr->meta();
    const std::size_t ingest_limit = std::min(limit, meta.count);
    const std::size_t padded = vectorcache::transform::padded_dim(meta.dim);
    const bool capture_vectors = show_index.has_value() && variance;

    std::cout << "Dataset: " << meta.label << " (dim=" << meta.dim << ", padded=" << padded
              << ", available=" << meta.count << ", ingesting=" << ingest_limit
              << ", srht_seed=" << seed << ", rounds=" << rounds << ")\n";

    VarianceHook variance_hook(capture_vectors);
    const auto ingest_start = std::chrono::steady_clock::now();

    std::vector<double> pre_variances;
    std::vector<std::vector<double>> round_variances;
    vectorcache::ingest::IngestReport report{};
    vectorcache::ingest::IngestionEngine engine =
        vectorcache::ingest::IngestionEngine::with_rotation(meta.dim, seed);

    if (rounds == 1) {
      LimitedReader limited(*reader_ptr, ingest_limit, variance ? &pre_variances : nullptr);
      engine.reserve_vectors(ingest_limit);
      if (variance) {
        report = engine.ingest_with_hook(limited, &variance_hook);
      } else {
        report = engine.ingest(limited);
      }
    } else {
      if (variance) {
        round_variances.resize(rounds);
        for (auto& rv : round_variances) rv.reserve(ingest_limit);
      }
      MultiRoundReader limited(*reader_ptr, ingest_limit, variance ? &pre_variances : nullptr,
                               rounds, seed, variance ? &round_variances : nullptr);
      std::size_t store_dim = meta.dim;
      for (std::size_t r = 0; r < rounds; ++r) {
        store_dim = vectorcache::transform::padded_dim(store_dim);
      }
      engine = vectorcache::ingest::IngestionEngine::from_rotated(store_dim);
      engine.reserve_vectors(ingest_limit);
      if (variance) {
        report = engine.ingest_with_hook(limited, &variance_hook);
      } else {
        report = engine.ingest(limited);
      }
    }

    const auto elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                                             ingest_start)
            .count());

    if (variance) {
      print_variance_stats("Pre-ingestion (raw)", pre_variances);
      for (std::size_t i = 0; i < round_variances.size(); ++i) {
        print_variance_stats("After SRHT round " + std::to_string(i + 1), round_variances[i]);
      }
      print_variance_stats("Post-ingestion (stored)", variance_hook.post_variances());
    }

    std::cout << "Ingested: " << report.vectors_ingested << " vectors\n";
    std::cout << "Total ingestion time: "
              << vectorcache::ingest::TimingSummary::format_duration(elapsed_ns) << " ("
              << elapsed_ns << " ns)\n";
    if (report.vectors_ingested > 0) {
      const double per_vec = static_cast<double>(elapsed_ns) / report.vectors_ingested;
      std::cout << "  per-vector: "
                << vectorcache::ingest::TimingSummary::format_duration(
                       static_cast<std::uint64_t>(per_vec))
                << " (" << per_vec << " ns)\n";
    }
    std::cout << "Blocks: " << report.full_blocks << " full, " << report.partial_len
              << " in partial block (L1 words/vec: " << engine.store().l1_words_per_vec()
              << ")\n";

    if (show_index) {
      if (*show_index >= report.vectors_ingested) {
        throw vectorcache::Error("--show-index out of range");
      }
      print_stored_l1_codes(*show_index, engine, padded);
      const auto& vectors = variance_hook.vectors();
      if (*show_index < vectors.size()) {
        const auto& vector = vectors[*show_index];
        double norm = 0.0;
        for (float x : vector) norm += static_cast<double>(x) * x;
        norm = std::sqrt(norm);
        std::cout << "Rotated f32 at index " << *show_index << " (dim=" << vector.size()
                  << ", L2 norm=" << std::fixed << std::setprecision(6) << norm << "):\n";
        const std::size_t preview = std::min<std::size_t>(16, vector.size());
        std::cout << "  [";
        for (std::size_t i = 0; i < preview; ++i) {
          if (i > 0) std::cout << ", ";
          std::cout << std::showpos << std::fixed << std::setprecision(6) << vector[i]
                    << std::noshowpos;
        }
        std::cout << "]\n";
        if (vector.size() > preview) {
          std::cout << "  ... (" << (vector.size() - preview) << " more dims)\n";
        }
      } else if (!variance) {
        std::cout << "(pass --variance with --show-index to also print the rotated f32 vector)\n";
      }
    }

    return 0;
  } catch (const vectorcache::Error& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
