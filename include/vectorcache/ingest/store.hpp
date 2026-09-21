#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/index/rp_buckets.hpp"

namespace vectorcache::ingest {

/// Flat in-RAM index: contiguous L0 codes + parallel ids + per-vector IP scales.
/// After single-table `finalize_buckets`, rows are sorted by that table's cell key.
/// Multi-table finalize keeps ingest order and stores per-table postings.
class VectorStore {
 public:
  VectorStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
              std::size_t bits_per_dim = 1, std::size_t block_dims = 1);
  static VectorStore with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                   std::size_t srht_dim, std::size_t vector_count,
                                   std::size_t bits_per_dim = 1, std::size_t block_dims = 1);

  std::size_t l0_words_per_vec() const { return l0_words_per_vec_; }
  /// Bits per codebook block (per dim when block_dims == 1).
  std::size_t bits_per_dim() const { return bits_per_dim_; }
  std::size_t block_dims() const { return block_dims_; }
  std::size_t input_dim() const { return input_dim_; }
  std::size_t srht_dim() const { return srht_dim_; }
  std::size_t size() const { return ids_.size(); }
  bool empty() const { return ids_.empty(); }

  std::span<const std::uint64_t> vector_l0(std::size_t index) const;
  std::span<const std::uint64_t> l0_codes() const { return codes_; }
  std::span<const std::size_t> ids() const { return ids_; }
  std::span<const float> scales() const { return scales_; }
  std::size_t id_at(std::size_t index) const;
  float scale_at(std::size_t index) const;

  void push(std::size_t id, std::span<const std::uint64_t> l0, float scale = 1.0f);
  void reserve(std::size_t n);

  /// Deep copy of codes/ids/scales/dims without bucket index (ingest order preserved).
  VectorStore clone() const;

  /// Permute rows by `order` (must be a permutation of [0, size())).
  void permute(std::span<const std::size_t> order);

  /// Single-table: argsort by cell keys, permute store, build contiguous CSR.
  void finalize_buckets(std::span<const std::uint64_t> cell_keys, index::ProjectionMatrix matrix,
                        index::BinCodec codec);

  /// Multi-table: `all_keys` is table-major (table t occupies [t*n, (t+1)*n)).
  /// T==1 uses permute fast path; T>1 keeps ingest order and builds postings per table.
  void finalize_buckets(std::span<const std::uint64_t> all_keys,
                        std::vector<index::ProjectionMatrix> matrices, index::BinCodec codec);

  bool has_buckets() const { return !tables_.empty(); }
  std::size_t num_tables() const { return tables_.size(); }
  const index::BucketIndex& buckets() const { return buckets(0); }
  const index::BucketIndex& buckets(std::size_t table) const;

 private:
  std::size_t l0_words_per_vec_;
  std::size_t bits_per_dim_;
  std::size_t block_dims_;
  std::size_t input_dim_;
  std::size_t srht_dim_;
  AlignedVector<std::uint64_t> codes_;
  std::vector<std::size_t> ids_;
  std::vector<float> scales_;
  std::vector<index::BucketIndex> tables_;
};

}  // namespace vectorcache::ingest
