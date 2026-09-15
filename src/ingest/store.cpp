#include "vectorcache/ingest/store.hpp"

#include <algorithm>
#include <cstring>

#include "vectorcache/error.hpp"

namespace vectorcache::ingest {

namespace {

std::size_t next_pow2(std::size_t n) {
  std::size_t p = 1;
  while (p < n) {
    p <<= 1;
  }
  return p;
}

}  // namespace

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

ParentStore::ParentStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
                         std::size_t top_d)
    : l0_words_per_vec_(l0_words_per_vec),
      input_dim_(input_dim),
      srht_dim_(srht_dim),
      top_d_(top_d) {
  if (l0_words_per_vec_ == 0 || input_dim_ == 0 || srht_dim_ == 0) {
    throw Error("ParentStore dimensions must be > 0");
  }
  if (top_d_ == 0 || top_d_ > quantize::kMaxSupportDepth) {
    throw Error("ParentStore top_d out of range");
  }
  if (top_d_ > input_dim_) {
    throw Error("ParentStore top_d exceeds input_dim");
  }
  if (srht_dim_ < input_dim_) {
    throw Error("ParentStore srht_dim must be >= input_dim");
  }
  slots_.assign(16, Slot{});
}

ParentStore ParentStore::with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                       std::size_t srht_dim, std::size_t top_d,
                                       std::size_t vector_count) {
  ParentStore store(l0_words_per_vec, input_dim, srht_dim, top_d);
  store.reserve_vectors(vector_count);
  return store;
}

void ParentStore::reserve_vectors(std::size_t vector_count) {
  // Pre-size only the open-address map. ParentGroup storage grows in push() when
  // a vector is posted under that key (top-d keys are often near-unique / O(N)).
  const std::size_t want_slots = next_pow2(std::max<std::size_t>(16, vector_count * 2));
  if (want_slots > slots_.size()) {
    rehash(want_slots);
  }
}

std::size_t ParentStore::probe_slot(const quantize::SupportKey& key) const {
  const std::size_t mask = slots_.size() - 1;
  std::size_t idx = static_cast<std::size_t>(quantize::support_key_hash(key)) & mask;
  while (slot_occupied(slots_[idx])) {
    if (slots_[idx].key == key) {
      return idx;
    }
    idx = (idx + 1) & mask;
  }
  return idx;
}

void ParentStore::insert_mapping(const quantize::SupportKey& key, std::size_t group_idx) {
  if ((map_size_ + 1) * 2 > slots_.size()) {
    rehash(slots_.size() * 2);
  }
  const std::size_t idx = probe_slot(key);
  if (!slot_occupied(slots_[idx])) {
    slots_[idx].key = key;
    slots_[idx].group_idx = group_idx;
    ++map_size_;
  } else {
    slots_[idx].group_idx = group_idx;
  }
}

void ParentStore::rehash(std::size_t new_cap) {
  new_cap = next_pow2(std::max<std::size_t>(16, new_cap));
  std::vector<Slot> old = std::move(slots_);
  slots_.assign(new_cap, Slot{});
  map_size_ = 0;
  for (const Slot& s : old) {
    if (!slot_occupied(s)) {
      continue;
    }
    const std::size_t mask = slots_.size() - 1;
    std::size_t idx = static_cast<std::size_t>(quantize::support_key_hash(s.key)) & mask;
    while (slot_occupied(slots_[idx])) {
      idx = (idx + 1) & mask;
    }
    slots_[idx] = s;
    ++map_size_;
  }
}

const ParentGroup* ParentStore::find(const quantize::SupportKey& key) const {
  if (slots_.empty()) {
    return nullptr;
  }
  const std::size_t mask = slots_.size() - 1;
  std::size_t idx = static_cast<std::size_t>(quantize::support_key_hash(key)) & mask;
  while (slot_occupied(slots_[idx])) {
    if (slots_[idx].key == key) {
      return &groups_[slots_[idx].group_idx];
    }
    idx = (idx + 1) & mask;
  }
  return nullptr;
}

const ParentGroup& ParentStore::group_at(std::size_t key_idx) const {
  if (key_idx >= groups_.size()) {
    throw Error("ParentStore::group_at index out of range");
  }
  return groups_[key_idx];
}

void ParentStore::push_vector(const quantize::SupportKey& key, std::span<const std::uint64_t> l0,
                              std::size_t id) {
  if (key.d != top_d_) {
    throw Error("SupportKey depth mismatch with store top_d");
  }
  // Grow before probe so a single walk resolves empty-or-hit.
  if ((map_size_ + 1) * 2 > slots_.size()) {
    rehash(slots_.size() * 2);
  }
  const std::size_t idx = probe_slot(key);
  std::size_t group_idx;
  if (!slot_occupied(slots_[idx])) {
    group_idx = groups_.size();
    keys_.push_back(key);
    groups_.emplace_back(l0_words_per_vec_);
    slots_[idx].key = key;
    slots_[idx].group_idx = group_idx;
    ++map_size_;
  } else {
    group_idx = slots_[idx].group_idx;
  }
  groups_[group_idx].push(id, l0);
  ++total_vectors_;
  invalidate_dim_postings();
}

void ParentStore::invalidate_dim_postings() const {
  dim_postings_.reset();
  dim_postings_key_count_ = 0;
}

const quantize::DimPostingIndex& ParentStore::dim_postings() const {
  if (!dim_postings_ || dim_postings_key_count_ != keys_.size()) {
    dim_postings_ = quantize::DimPostingIndex::build(keys_, input_dim_);
    dim_postings_key_count_ = keys_.size();
  }
  return *dim_postings_;
}

}  // namespace vectorcache::ingest
