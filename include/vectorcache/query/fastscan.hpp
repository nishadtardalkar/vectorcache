#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/aligned.hpp"

namespace vectorcache::query {

/// FAISS FastScan-style blocked code cache for flat scan.
///
/// Byte layout (bits 1/2/4/8): for each block of kBlock vectors, for each
/// byte-group g, store kBlock consecutive bytes (one per vector in the block).
/// Index: ((block * num_groups) + group) * kBlock + vec_in_block.
///
/// Bits=1 also keeps a transposed u64 view for AVX-512 mask-add:
/// Index: ((block * words_per_vec) + word) * kBlock + vec_in_block.
class BlockedCodes {
 public:
  static constexpr std::size_t kBlock = 32;

  BlockedCodes() = default;

  bool empty() const { return n_ == 0; }
  std::size_t size() const { return n_; }
  std::size_t bits() const { return bits_; }
  std::size_t dim() const { return dim_; }
  std::size_t num_groups() const { return num_groups_; }
  std::size_t words_per_vec() const { return words_per_vec_; }
  std::size_t n_blocks() const { return n_blocks_; }
  bool has_bit1_words() const { return !bit1_words_.empty(); }

  /// Rebuild from vector-major packed codes. Pads the last block with zeros.
  /// `dim` is srht_dim; `block_dims` is codebook block size (default 1).
  void rebuild(std::span<const std::uint64_t> codes, std::size_t words_per_vec, std::size_t n,
               std::size_t dim, std::size_t bits, std::size_t block_dims = 1);

  void clear();

  std::size_t block_dims() const { return block_dims_; }
  std::size_t num_codes() const { return num_codes_; }

  /// Pointer to kBlock bytes for (block, group).
  const std::uint8_t* group_bytes(std::size_t block, std::size_t group) const;

  /// Pointer to kBlock u64 words for (block, word_index); bits=1 only.
  const std::uint64_t* bit1_word_column(std::size_t block, std::size_t word) const;

  /// Valid vector count in a block (last block may be partial).
  std::size_t block_count(std::size_t block) const;

  /// Unpack one vector's packed bytes from the blocked layout (for tests).
  void unpack_vector_bytes(std::size_t index, std::span<std::uint8_t> out) const;

 private:
  std::size_t n_ = 0;
  std::size_t bits_ = 0;
  std::size_t dim_ = 0;
  std::size_t block_dims_ = 1;
  std::size_t num_codes_ = 0;
  std::size_t num_groups_ = 0;
  std::size_t words_per_vec_ = 0;
  std::size_t n_blocks_ = 0;
  AlignedVector<std::uint8_t> bytes_;
  AlignedVector<std::uint64_t> bit1_words_;
};

}  // namespace vectorcache::query
