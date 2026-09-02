#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/ingest/block.hpp"

namespace vectorcache::ingest {

/// In-memory store of full L1 bitcode blocks plus one in-progress partial block.
class BlockStore {
 public:
  explicit BlockStore(std::size_t l1_words_per_vec);
  static BlockStore with_capacity(std::size_t l1_words_per_vec, std::size_t vector_count);

  std::size_t l1_words_per_vec() const { return l1_words_per_vec_; }
  std::size_t block_count() const { return blocks_.size(); }
  std::size_t total_vectors() const;

  const VectorBlock* get_block(std::size_t index) const;
  const VectorBlock& partial_block() const { return partial_; }
  const std::vector<VectorBlock>& blocks() const { return blocks_; }

  void push_l1_codes(std::span<const std::uint64_t> codes);

 private:
  std::size_t l1_words_per_vec_;
  std::vector<VectorBlock> blocks_;
  VectorBlock partial_;
};

}  // namespace vectorcache::ingest
