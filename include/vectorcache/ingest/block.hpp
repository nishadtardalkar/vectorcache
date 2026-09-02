#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorcache::ingest {

inline constexpr std::size_t BLOCK_SIZE = 1024;

/// Contiguous storage for up to BLOCK_SIZE L1 bitcode vectors.
class VectorBlock {
 public:
  explicit VectorBlock(std::size_t l1_words_per_vec);
  VectorBlock(std::size_t l1_words_per_vec, std::size_t vector_capacity);

  std::size_t l1_words_per_vec() const { return l1_words_per_vec_; }
  std::size_t len() const { return len_; }
  bool is_empty() const { return len_ == 0; }
  bool is_full() const { return len_ == BLOCK_SIZE; }

  void push_l1(std::span<const std::uint64_t> codes);
  std::span<const std::uint64_t> as_slice() const;
  void reset();
  VectorBlock take_full();

 private:
  std::size_t l1_words_per_vec_;
  std::vector<std::uint64_t> data_;
  std::size_t len_;
};

}  // namespace vectorcache::ingest
