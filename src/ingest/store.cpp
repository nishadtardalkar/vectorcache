#include "vectorcache/ingest/store.hpp"

#include <algorithm>
#include <cstring>

#include "vectorcache/error.hpp"

namespace vectorcache::ingest {

ParentGroup::ParentGroup(std::size_t l0_words_per_vec) : l0_words_per_vec_(l0_words_per_vec) {
  if (l0_words_per_vec_ == 0) {
    throw Error("ParentGroup requires l0_words_per_vec > 0");
  }
}

std::span<const std::uint64_t> ParentGroup::vector_l0(std::size_t index) const {
  if (index >= ids_.size()) {
    throw Error("ParentGroup::vector_l0 index out of range");
  }
  return {codes_.data() + index * l0_words_per_vec_, l0_words_per_vec_};
}

void ParentGroup::reserve(std::size_t n) {
  codes_.reserve(n * l0_words_per_vec_);
  ids_.reserve(n);
}

void ParentGroup::push(std::size_t id, std::span<const std::uint64_t> l0) {
  if (l0.size() != l0_words_per_vec_) {
    throw Error("L0 word count mismatch: expected " + std::to_string(l0_words_per_vec_) +
                ", got " + std::to_string(l0.size()));
  }
  const std::size_t offset = codes_.size();
  codes_.resize(offset + l0_words_per_vec_);
  std::memcpy(codes_.data() + offset, l0.data(), l0.size_bytes());
  ids_.push_back(id);
}

ParentStore::ParentStore(std::size_t l0_words_per_vec, std::size_t padded_dim)
    : l0_words_per_vec_(l0_words_per_vec), padded_dim_(padded_dim) {
  if (l0_words_per_vec_ == 0 || padded_dim_ == 0) {
    throw Error("ParentStore dimensions must be > 0");
  }
  if (padded_dim_ % quantize::PARENT_BITS != 0) {
    throw Error("ParentStore padded_dim must be divisible by 8");
  }
  by_key_.fill(kInvalidGroup);
  keys_.reserve(kMaxParents);
  groups_.reserve(kMaxParents);
}

ParentStore ParentStore::with_capacity(std::size_t l0_words_per_vec, std::size_t padded_dim,
                                       std::size_t vector_count) {
  ParentStore store(l0_words_per_vec, padded_dim);
  store.reserve_vectors(vector_count);
  return store;
}

void ParentStore::reserve_vectors(std::size_t vector_count) {
  // Prefer a larger first-touch hint so push avoids repeated realloc+memcpy.
  // Occupancy is skewed across parents; /8 is a compromise vs /16.
  reserve_hint_ = std::max<std::size_t>(1, vector_count / 8);
  for (auto& group : groups_) {
    group.reserve(reserve_hint_);
  }
}

const ParentGroup* ParentStore::group_for_key(std::uint8_t key) const {
  const std::size_t idx = by_key_[key];
  if (idx == kInvalidGroup) {
    return nullptr;
  }
  return &groups_[idx];
}

void ParentStore::push_vector(std::uint8_t parent_key, std::span<const std::uint64_t> l0,
                              std::size_t id) {
  std::size_t idx = by_key_[parent_key];
  if (idx == kInvalidGroup) {
    idx = groups_.size();
    by_key_[parent_key] = idx;
    keys_.push_back(parent_key);
    groups_.emplace_back(l0_words_per_vec_);
    if (reserve_hint_ > 0) {
      groups_[idx].reserve(reserve_hint_);
    }
  }
  groups_[idx].push(id, l0);
  ++total_vectors_;
}

}  // namespace vectorcache::ingest
