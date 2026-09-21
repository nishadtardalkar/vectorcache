#include "vectorcache/ingest/store.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::ingest {

VectorStore::VectorStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
                         std::size_t bits_per_dim, std::size_t block_dims)
    : l0_words_per_vec_(l0_words_per_vec),
      bits_per_dim_(bits_per_dim),
      block_dims_(block_dims),
      input_dim_(input_dim),
      srht_dim_(srht_dim) {
  if (l0_words_per_vec_ == 0 || input_dim_ == 0 || srht_dim_ == 0) {
    throw Error("VectorStore dimensions must be > 0");
  }
  if (srht_dim_ < input_dim_) {
    throw Error("VectorStore srht_dim must be >= input_dim");
  }
  quantize::validate_quant_spec(srht_dim_, quantize::QuantSpec{block_dims_, bits_per_dim_});
  const std::size_t expected = quantize::l0_words_per_vector(srht_dim_, bits_per_dim_, block_dims_);
  if (l0_words_per_vec_ != expected) {
    throw Error("VectorStore l0_words_per_vec mismatch for bits_per_dim/block_dims");
  }
}

VectorStore VectorStore::with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                       std::size_t srht_dim, std::size_t vector_count,
                                       std::size_t bits_per_dim, std::size_t block_dims) {
  VectorStore store(l0_words_per_vec, input_dim, srht_dim, bits_per_dim, block_dims);
  store.reserve(vector_count);
  return store;
}

std::span<const std::uint64_t> VectorStore::vector_l0(std::size_t index) const {
  if (index >= ids_.size()) {
    throw Error("VectorStore::vector_l0 index out of range");
  }
  return {codes_.data() + index * l0_words_per_vec_, l0_words_per_vec_};
}

std::size_t VectorStore::id_at(std::size_t index) const {
  if (index >= ids_.size()) {
    throw Error("VectorStore::id_at index out of range");
  }
  return ids_[index];
}

float VectorStore::scale_at(std::size_t index) const {
  if (index >= scales_.size()) {
    throw Error("VectorStore::scale_at index out of range");
  }
  return scales_[index];
}

void VectorStore::reserve(std::size_t n) {
  codes_.reserve(n * l0_words_per_vec_);
  ids_.reserve(n);
  scales_.reserve(n);
}

VectorStore VectorStore::clone() const {
  VectorStore out(l0_words_per_vec_, input_dim_, srht_dim_, bits_per_dim_, block_dims_);
  out.codes_ = codes_;
  out.ids_ = ids_;
  out.scales_ = scales_;
  return out;
}

void VectorStore::push(std::size_t id, std::span<const std::uint64_t> l0, float scale) {
  if (l0.size() != l0_words_per_vec_) {
    throw Error("L0 word count mismatch: expected " + std::to_string(l0_words_per_vec_) + ", got " +
                std::to_string(l0.size()));
  }
  if (buckets_.has_value()) {
    throw Error("VectorStore::push after finalize_buckets is not supported");
  }
  const std::size_t offset = codes_.size();
  codes_.resize(offset + l0_words_per_vec_);
  std::memcpy(codes_.data() + offset, l0.data(), l0.size_bytes());
  ids_.push_back(id);
  scales_.push_back(scale);
}

void VectorStore::permute(std::span<const std::size_t> order) {
  const std::size_t n = ids_.size();
  if (order.size() != n) {
    throw Error("VectorStore::permute order size mismatch");
  }
  if (n == 0) {
    return;
  }

  std::vector<unsigned char> seen(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t j = order[i];
    if (j >= n || seen[j]) {
      throw Error("VectorStore::permute order is not a permutation");
    }
    seen[j] = 1;
  }

  AlignedVector<std::uint64_t> new_codes(n * l0_words_per_vec_);
  std::vector<std::size_t> new_ids(n);
  std::vector<float> new_scales(n);
  for (std::size_t new_i = 0; new_i < n; ++new_i) {
    const std::size_t old_i = order[new_i];
    std::memcpy(new_codes.data() + new_i * l0_words_per_vec_,
                codes_.data() + old_i * l0_words_per_vec_, l0_words_per_vec_ * sizeof(std::uint64_t));
    new_ids[new_i] = ids_[old_i];
    new_scales[new_i] = scales_[old_i];
  }
  codes_ = std::move(new_codes);
  ids_ = std::move(new_ids);
  scales_ = std::move(new_scales);
  buckets_.reset();
}

void VectorStore::finalize_buckets(std::span<const std::uint64_t> cell_keys, index::PairHash hash,
                                   float bin_width) {
  const std::size_t n = ids_.size();
  if (cell_keys.size() != n) {
    throw Error("finalize_buckets: cell_keys size mismatch");
  }
  if (n == 0) {
    throw Error("finalize_buckets: empty store");
  }
  if (hash.empty()) {
    throw Error("finalize_buckets: empty PairHash");
  }

  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (cell_keys[a] != cell_keys[b]) {
      return cell_keys[a] < cell_keys[b];
    }
    return a < b;
  });

  std::vector<std::uint64_t> sorted_keys(n);
  for (std::size_t i = 0; i < n; ++i) {
    sorted_keys[i] = cell_keys[order[i]];
  }

  permute(order);
  buckets_ = index::BucketIndex::build(sorted_keys, std::move(hash), bin_width);
}

const index::BucketIndex& VectorStore::buckets() const {
  if (!buckets_.has_value()) {
    throw Error("VectorStore::buckets: no bucket index");
  }
  return *buckets_;
}

}  // namespace vectorcache::ingest
