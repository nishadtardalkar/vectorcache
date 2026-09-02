#include "vectorcache/ingest/store.hpp"

namespace vectorcache::ingest {

BlockStore::BlockStore(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                       std::size_t padded_dim)
    : layout_(make_block_layout(l1_words_per_vec, l0_words_per_vec, padded_dim)),
      partial_(l1_words_per_vec, l0_words_per_vec, padded_dim) {}

BlockStore BlockStore::with_capacity(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                                     std::size_t padded_dim, std::size_t vector_count) {
  BlockStore store(l1_words_per_vec, l0_words_per_vec, padded_dim);
  const std::size_t full_blocks = vector_count / BLOCK_SIZE;
  const std::size_t partial_reserve = std::max(vector_count % BLOCK_SIZE, std::size_t{1});
  store.blocks_.reserve(full_blocks);
  store.partial_ = LogicalBlock(l1_words_per_vec, l0_words_per_vec, padded_dim, partial_reserve);
  return store;
}

std::size_t BlockStore::total_vectors() const {
  std::size_t total = partial_.len();
  for (const auto& block : blocks_) {
    total += block.len();
  }
  return total;
}

const LogicalBlock* BlockStore::get_block(std::size_t index) const {
  if (index >= blocks_.size()) {
    return nullptr;
  }
  return &blocks_[index];
}

void BlockStore::push_vector(std::span<const std::uint64_t> l1, std::span<const std::uint64_t> l0,
                             std::span<const float> full) {
  partial_.push(l1, l0, full);
  if (partial_.is_full()) {
    blocks_.push_back(partial_.take_full());
  }
}

}  // namespace vectorcache::ingest
