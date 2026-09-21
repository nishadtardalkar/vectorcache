#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/srht.hpp"

namespace vectorcache::ingest {

inline constexpr std::size_t INGEST_BATCH_SIZE = 256;

struct IngestReport {
  std::uint64_t vectors_ingested = 0;
};

struct BucketParams {
  std::size_t num_pair_dirs = 8;
  float bin_width = 0.1f;
  std::uint64_t bucket_seed = 0;  // 0 => derive from rotation seed when available
};

class IngestionEngine {
 public:
  /// Already-SRHT'd vectors of length srht_dim (input_dim == srht_dim).
  static IngestionEngine from_rotated(std::size_t srht_dim, std::size_t bits_per_dim = 1,
                                      std::size_t block_dims = 1,
                                      BucketParams buckets = BucketParams{});
  static IngestionEngine with_rotation(std::size_t original_dim, std::uint64_t seed,
                                       std::size_t bits_per_dim = 1, std::size_t block_dims = 1,
                                       BucketParams buckets = BucketParams{});

  void reserve_vectors(std::size_t count);
  /// When `finalize_buckets` is false, codes are stored but pair-hash CSR is not built.
  IngestReport ingest(datasets::DatasetReader& reader, bool finalize_buckets = true);
  IngestReport ingest_with_hook(datasets::DatasetReader& reader, VectorHook* hook,
                                bool finalize_buckets = true);
  const VectorStore& store() const { return store_; }
  const quantize::LloydMaxCodebook& codebook() const { return codebook_; }
  std::size_t bits_per_dim() const { return codebook_.bits(); }
  std::size_t block_dims() const { return codebook_.block_dims(); }
  const BucketParams& bucket_params() const { return bucket_params_; }
  const index::PairHash& pair_hash() const { return pair_hash_; }

 private:
  struct VectorWork {
    AlignedVector<float> buf;  // length srht_dim_
    AlignedVector<std::uint64_t> l0;
    float alpha = 1.0f;
    std::uint64_t cell_key = 0;
  };

  IngestionEngine(VectorStore store, std::optional<transform::SrhtRotation> rotation,
                  bool quantize_only, std::size_t input_dim, std::size_t srht_dim,
                  std::size_t l0_words_per_vec, quantize::LloydMaxCodebook codebook,
                  BucketParams bucket_params, std::uint64_t resolved_bucket_seed);

  void ensure_batch_capacity(std::size_t batch_cap);
  std::size_t read_batch(datasets::DatasetReader& reader);
  void process_batch(std::size_t batch_len);
  void finalize_bucket_index(std::span<const std::uint64_t> all_keys);

  VectorStore store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool quantize_only_;
  std::size_t input_dim_;
  std::size_t srht_dim_;
  std::size_t l0_words_per_vec_;
  quantize::LloydMaxCodebook codebook_;
  BucketParams bucket_params_;
  std::uint64_t bucket_seed_;
  index::PairHash pair_hash_;
  std::vector<VectorWork> batch_work_;
};

}  // namespace vectorcache::ingest
