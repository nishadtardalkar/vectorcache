#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/aligned.hpp"

namespace vectorcache::ingest {

inline constexpr std::size_t BLOCK_SIZE = 1024;

/// Byte layout for a single logical block (one contiguous allocation).
struct BlockLayout {
  std::size_t l1_words_per_vec = 0;
  std::size_t l0_words_per_vec = 0;
  std::size_t padded_dim = 0;
  std::size_t phys_block_bytes = 0;
  std::size_t l0_phys_blocks = 0;
  std::size_t full_phys_blocks = 0;
  std::size_t l0_vecs_per_sub = 0;
  std::size_t full_vecs_per_sub = 0;
  std::size_t total_bytes = 0;

  std::size_t l1_region_offset() const { return 0; }
  std::size_t l0_region_offset() const { return phys_block_bytes; }
  std::size_t full_region_offset() const {
    return phys_block_bytes * (1 + l0_phys_blocks);
  }
  std::size_t l0_sub_offset(std::size_t sub_idx) const {
    return l0_region_offset() + sub_idx * phys_block_bytes;
  }
  std::size_t full_sub_offset(std::size_t sub_idx) const {
    return full_region_offset() + sub_idx * phys_block_bytes;
  }
  std::size_t vector_l1_offset(std::size_t vec_idx) const {
    return l1_region_offset() + vec_idx * l1_words_per_vec * sizeof(std::uint64_t);
  }
  std::size_t vector_l0_offset(std::size_t vec_idx) const {
    const std::size_t sub = vec_idx / l0_vecs_per_sub;
    const std::size_t in_sub = vec_idx % l0_vecs_per_sub;
    return l0_sub_offset(sub) + in_sub * l0_words_per_vec * sizeof(std::uint64_t);
  }
  std::size_t vector_full_offset(std::size_t vec_idx) const {
    const std::size_t sub = vec_idx / full_vecs_per_sub;
    const std::size_t in_sub = vec_idx % full_vecs_per_sub;
    return full_sub_offset(sub) + in_sub * padded_dim * sizeof(float);
  }
};

BlockLayout make_block_layout(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                              std::size_t padded_dim);

/// Single-buffer storage for up to BLOCK_SIZE vectors across L1, L0, and full tiers.
class LogicalBlock {
 public:
  LogicalBlock(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec, std::size_t padded_dim);
  LogicalBlock(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec, std::size_t padded_dim,
               std::size_t /*vector_capacity*/);

  const BlockLayout& layout() const { return layout_; }
  std::size_t len() const { return len_; }
  bool is_empty() const { return len_ == 0; }
  bool is_full() const { return len_ == BLOCK_SIZE; }

  void push(std::span<const std::uint64_t> l1, std::span<const std::uint64_t> l0,
            std::span<const float> full);
  std::span<const std::uint64_t> l1_slice() const;
  std::span<const std::uint64_t> l0_slice() const;
  std::span<const float> full_slice() const;
  std::span<const std::uint64_t> as_slice() const { return l1_slice(); }
  std::span<const std::byte> bytes() const;

  void reset();
  LogicalBlock take_full();

 private:
  BlockLayout layout_;
  AlignedVector<std::byte> data_;
  std::size_t len_;
};

using VectorBlock = LogicalBlock;

}  // namespace vectorcache::ingest
