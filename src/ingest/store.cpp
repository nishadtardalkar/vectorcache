#include "vectorcache/ingest/store.hpp"

#include <cstring>
#include <string>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::ingest {

VectorStore::VectorStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
                         std::size_t bits_per_dim)
    : l0_words_per_vec_(l0_words_per_vec),
      bits_per_dim_(bits_per_dim),
      input_dim_(input_dim),
      srht_dim_(srht_dim) {
  if (l0_words_per_vec_ == 0 || input_dim_ == 0 || srht_dim_ == 0) {
    throw Error("VectorStore dimensions must be > 0");
  }
  if (srht_dim_ < input_dim_) {
    throw Error("VectorStore srht_dim must be >= input_dim");
  }
  quantize::validate_bits_per_dim(bits_per_dim_);
  const std::size_t expected = quantize::l0_words_per_vector(srht_dim_, bits_per_dim_);
  if (l0_words_per_vec_ != expected) {
    throw Error("VectorStore l0_words_per_vec mismatch for bits_per_dim");
  }
}

VectorStore VectorStore::with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                       std::size_t srht_dim, std::size_t vector_count,
                                       std::size_t bits_per_dim) {
  VectorStore store(l0_words_per_vec, input_dim, srht_dim, bits_per_dim);
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

void VectorStore::reserve(std::size_t n) {
  codes_.reserve(n * l0_words_per_vec_);
  ids_.reserve(n);
}

void VectorStore::push(std::size_t id, std::span<const std::uint64_t> l0) {
  if (l0.size() != l0_words_per_vec_) {
    throw Error("L0 word count mismatch: expected " + std::to_string(l0_words_per_vec_) + ", got " +
                std::to_string(l0.size()));
  }
  const std::size_t offset = codes_.size();
  codes_.resize(offset + l0_words_per_vec_);
  std::memcpy(codes_.data() + offset, l0.data(), l0.size_bytes());
  ids_.push_back(id);
}

}  // namespace vectorcache::ingest
