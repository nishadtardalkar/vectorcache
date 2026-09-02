#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/ingest/block.hpp"

namespace vectorcache::ingest {

/// In-memory store of full tiered blocks plus one in-progress partial block.
class BlockStore {
 public:
  BlockStore(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec, std::size_t padded_dim);
  static BlockStore with_capacity(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                                  std::size_t padded_dim, std::size_t vector_count);

  std::size_t l1_words_per_vec() const { return layout_.l1_words_per_vec; }
  std::size_t l0_words_per_vec() const { return layout_.l0_words_per_vec; }
  std::size_t padded_dim() const { return layout_.padded_dim; }
  const BlockLayout& layout() const { return layout_; }
  std::size_t block_count() const { return blocks_.size(); }
  std::size_t total_vectors() const;

  const LogicalBlock* get_block(std::size_t index) const;
  const LogicalBlock& partial_block() const { return partial_; }
  const std::vector<LogicalBlock>& blocks() const { return blocks_; }

  void push_vector(std::span<const std::uint64_t> l1, std::span<const std::uint64_t> l0,
                   std::span<const float> full);

 private:
  BlockLayout layout_;
  std::vector<LogicalBlock> blocks_;
  LogicalBlock partial_;
};

}  // namespace vectorcache::ingest
