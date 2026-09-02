#include "vectorcache/ingest/block.hpp"

#include <algorithm>

#include "vectorcache/error.hpp"

namespace vectorcache::ingest {

VectorBlock::VectorBlock(std::size_t l1_words_per_vec)
    : l1_words_per_vec_(l1_words_per_vec), len_(0) {
  data_.reserve(BLOCK_SIZE * l1_words_per_vec_);
}

VectorBlock::VectorBlock(std::size_t l1_words_per_vec, std::size_t vector_capacity)
    : l1_words_per_vec_(l1_words_per_vec), len_(0) {
  const std::size_t cap = std::min(vector_capacity, BLOCK_SIZE);
  data_.reserve(cap * l1_words_per_vec_);
}

void VectorBlock::push_l1(std::span<const std::uint64_t> codes) {
  if (codes.size() != l1_words_per_vec_) {
    throw Error("L1 word count mismatch: expected " + std::to_string(l1_words_per_vec_) +
                ", got " + std::to_string(codes.size()));
  }
  if (is_full()) {
    throw Error("block is already full (" + std::to_string(BLOCK_SIZE) + " vectors)");
  }
  const std::size_t offset = len_ * l1_words_per_vec_;
  if (offset + l1_words_per_vec_ > data_.size()) {
    data_.resize(offset + l1_words_per_vec_, 0);
  }
  std::copy(codes.begin(), codes.end(), data_.begin() + static_cast<std::ptrdiff_t>(offset));
  ++len_;
}

std::span<const std::uint64_t> VectorBlock::as_slice() const {
  return {data_.data(), len_ * l1_words_per_vec_};
}

void VectorBlock::reset() {
  data_.clear();
  len_ = 0;
}

VectorBlock VectorBlock::take_full() {
  VectorBlock replacement(l1_words_per_vec_);
  replacement.data_.reserve(BLOCK_SIZE * l1_words_per_vec_);
  std::swap(data_, replacement.data_);
  const std::size_t taken_len = len_;
  len_ = 0;
  replacement.len_ = taken_len;
  return replacement;
}

}  // namespace vectorcache::ingest
