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
  /// Split a cell when its online count exceeds this (must be >= 1).
  std::size_t max_bucket_items = 1024;
  std::uint64_t bucket_seed = 0;  // reserved; first centroid is data-driven
};

class IngestionEngine {
 public:
  /// Already-SRHT'd vectors of length srht_dim (input_dim == srht_dim).
  static IngestionEngine from_rotated(std::size_t srht_dim, std::size_t bits_per_dim = 1,
                                      BucketParams buckets = BucketParams{});
  static IngestionEngine with_rotation(std::size_t original_dim, std::uint64_t seed,
                                       std::size_t bits_per_dim = 1,
                                       BucketParams buckets = BucketParams{});

  void reserve_vectors(std::size_t count);
  /// When `finalize_buckets` is false, codes are stored but cluster CSR is not built.
  IngestReport ingest(datasets::DatasetReader& reader, bool finalize_buckets = true);
  IngestReport ingest_with_hook(datasets::DatasetReader& reader, VectorHook* hook,
                                bool finalize_buckets = true);
  const VectorStore& store() const { return store_; }
  const quantize::LloydMaxCodebook& codebook() const { return codebook_; }
  const index::ClusterCentroids& centroids() const { return centroids_; }
  std::size_t bits_per_dim() const { return codebook_.bits(); }

 private:
  struct VectorWork {
    AlignedVector<float> buf;  // length srht_dim_
    AlignedVector<std::uint64_t> l0;
    float alpha = 1.0f;
  };

  IngestionEngine(VectorStore store, std::optional<transform::SrhtRotation> rotation,
                  bool quantize_only, std::size_t input_dim, std::size_t srht_dim,
                  std::size_t l0_words_per_vec, quantize::LloydMaxCodebook codebook,
                  BucketParams bucket_params);

  void ensure_batch_capacity(std::size_t batch_cap);
  std::size_t read_batch(datasets::DatasetReader& reader);
  void process_batch(std::size_t batch_len);
  void maybe_split(std::vector<std::uint64_t>& cell_keys);
  void finalize_bucket_index(std::vector<std::uint64_t>& cell_keys);

  VectorStore store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool quantize_only_;
  std::size_t input_dim_;
  std::size_t srht_dim_;
  std::size_t l0_words_per_vec_;
  quantize::LloydMaxCodebook codebook_;
  BucketParams bucket_params_;
  index::ClusterCentroids centroids_;
  /// Row-major post-SRHT floats retained for split membership (N * srht_dim).
  AlignedVector<float> rotated_all_;
  std::vector<VectorWork> batch_work_;
};

}  // namespace vectorcache::ingest
