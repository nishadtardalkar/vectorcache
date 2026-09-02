#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/transform/srht.hpp"

namespace vectorcache::ingest {

inline constexpr std::size_t INGEST_BATCH_SIZE = 256;

struct IngestReport {
  std::size_t full_blocks = 0;
  std::size_t partial_len = 0;
  std::uint64_t vectors_ingested = 0;
};

class IngestionEngine {
 public:
  static IngestionEngine from_rotated(std::size_t padded_dim);
  static IngestionEngine with_rotation(std::size_t original_dim, std::uint64_t seed);

  void reserve_vectors(std::size_t count);
  const transform::SrhtRotation* rotation() const;
  void push_l1_codes(std::span<const std::uint64_t> codes);
  IngestReport ingest(datasets::DatasetReader& reader);
  IngestReport ingest_with_hook(datasets::DatasetReader& reader, VectorHook* hook);
  const BlockStore& store() const { return store_; }

 private:
  struct VectorWork {
    AlignedVector<float> rotated;
    AlignedVector<std::uint64_t> l1;
  };

  IngestionEngine(BlockStore store, std::optional<transform::SrhtRotation> rotation,
                  bool quantize_only, std::size_t input_dim, std::size_t padded_dim,
                  std::size_t l1_words_per_vec);

  void ensure_batch_capacity(std::size_t batch_cap);
  std::size_t read_batch(datasets::DatasetReader& reader);
  void process_batch_parallel(std::size_t batch_len);

  BlockStore store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool quantize_only_;
  std::size_t input_dim_;
  std::size_t padded_dim_;
  std::size_t l1_words_per_vec_;
  std::vector<float> read_buf_;
  std::vector<float> batch_inputs_;
  std::vector<VectorWork> batch_work_;
};

}  // namespace vectorcache::ingest
