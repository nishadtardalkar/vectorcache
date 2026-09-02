#include "vectorcache/ingest/block.hpp"

#include <algorithm>
#include <cstring>

#include "vectorcache/error.hpp"

namespace vectorcache::ingest {

namespace {

std::size_t checked_div(std::size_t numerator, std::size_t denominator, const char* label) {
  if (denominator == 0 || numerator % denominator != 0) {
    throw Error(std::string("block layout: ") + label + " is not an integer ratio");
  }
  return numerator / denominator;
}

}  // namespace

BlockLayout make_block_layout(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                              std::size_t padded_dim) {
  if (l1_words_per_vec == 0 || l0_words_per_vec == 0 || padded_dim == 0) {
    throw Error("block layout dimensions must be > 0");
  }

  BlockLayout layout;
  layout.l1_words_per_vec = l1_words_per_vec;
  layout.l0_words_per_vec = l0_words_per_vec;
  layout.padded_dim = padded_dim;

  const std::size_t l1_bytes = l1_words_per_vec * sizeof(std::uint64_t);
  const std::size_t l0_bytes = l0_words_per_vec * sizeof(std::uint64_t);
  const std::size_t full_bytes = padded_dim * sizeof(float);

  layout.phys_block_bytes = BLOCK_SIZE * l1_bytes;
  layout.l0_phys_blocks = checked_div(l0_bytes, l1_bytes, "l0/l1 bytes per vector");
  layout.full_phys_blocks = checked_div(full_bytes, l1_bytes, "full/l1 bytes per vector");
  layout.l0_vecs_per_sub = checked_div(BLOCK_SIZE, layout.l0_phys_blocks, "vectors per l0 sub-block");
  layout.full_vecs_per_sub =
      checked_div(BLOCK_SIZE, layout.full_phys_blocks, "vectors per full sub-block");
  layout.total_bytes =
      layout.phys_block_bytes * (1 + layout.l0_phys_blocks + layout.full_phys_blocks);
  return layout;
}

LogicalBlock::LogicalBlock(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                           std::size_t padded_dim)
    : layout_(make_block_layout(l1_words_per_vec, l0_words_per_vec, padded_dim)), len_(0) {
  data_.assign(layout_.total_bytes, std::byte{0});
}

LogicalBlock::LogicalBlock(std::size_t l1_words_per_vec, std::size_t l0_words_per_vec,
                           std::size_t padded_dim, std::size_t /*vector_capacity*/)
    : LogicalBlock(l1_words_per_vec, l0_words_per_vec, padded_dim) {}

void LogicalBlock::push(std::span<const std::uint64_t> l1, std::span<const std::uint64_t> l0,
                        std::span<const float> full) {
  if (l1.size() != layout_.l1_words_per_vec) {
    throw Error("L1 word count mismatch: expected " + std::to_string(layout_.l1_words_per_vec) +
                ", got " + std::to_string(l1.size()));
  }
  if (l0.size() != layout_.l0_words_per_vec) {
    throw Error("L0 word count mismatch: expected " + std::to_string(layout_.l0_words_per_vec) +
                ", got " + std::to_string(l0.size()));
  }
  if (full.size() != layout_.padded_dim) {
    throw Error("full vector dimension mismatch: expected " + std::to_string(layout_.padded_dim) +
                ", got " + std::to_string(full.size()));
  }
  if (is_full()) {
    throw Error("block is already full (" + std::to_string(BLOCK_SIZE) + " vectors)");
  }

  std::byte* base = data_.data();
  std::memcpy(base + layout_.vector_l1_offset(len_), l1.data(), l1.size_bytes());
  std::memcpy(base + layout_.vector_l0_offset(len_), l0.data(), l0.size_bytes());
  std::memcpy(base + layout_.vector_full_offset(len_), full.data(), full.size_bytes());
  ++len_;
}

std::span<const std::uint64_t> LogicalBlock::l1_slice() const {
  if (len_ == 0) {
    return {};
  }
  const auto* base = reinterpret_cast<const std::uint64_t*>(data_.data());
  const std::size_t words = len_ * layout_.l1_words_per_vec;
  return {base, words};
}

std::span<const std::uint64_t> LogicalBlock::l0_slice() const {
  if (len_ == 0) {
    return {};
  }
  const auto* l0_base =
      reinterpret_cast<const std::uint64_t*>(data_.data() + layout_.l0_region_offset());
  const std::size_t words = len_ * layout_.l0_words_per_vec;
  return {l0_base, words};
}

std::span<const float> LogicalBlock::full_slice() const {
  if (len_ == 0) {
    return {};
  }
  const auto* full_base =
      reinterpret_cast<const float*>(data_.data() + layout_.full_region_offset());
  const std::size_t floats = len_ * layout_.padded_dim;
  return {full_base, floats};
}

std::span<const std::byte> LogicalBlock::bytes() const { return {data_.data(), data_.size()}; }

void LogicalBlock::reset() {
  std::fill(data_.begin(), data_.end(), std::byte{0});
  len_ = 0;
}

LogicalBlock LogicalBlock::take_full() {
  LogicalBlock replacement(layout_.l1_words_per_vec, layout_.l0_words_per_vec, layout_.padded_dim);
  std::swap(data_, replacement.data_);
  const std::size_t taken_len = len_;
  len_ = 0;
  replacement.len_ = taken_len;
  return replacement;
}

}  // namespace vectorcache::ingest
