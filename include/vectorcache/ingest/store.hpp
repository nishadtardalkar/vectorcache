#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::ingest {

/// Contiguous L0 codes + explicit ids for one parent key.
class ParentGroup {
 public:
  explicit ParentGroup(std::size_t l0_words_per_vec);

  std::size_t size() const { return ids_.size(); }
  bool empty() const { return ids_.empty(); }
  std::size_t id_at(std::size_t index) const { return ids_[index]; }

  std::span<const std::uint64_t> vector_l0(std::size_t index) const;
  std::span<const std::uint64_t> l0_codes() const { return codes_; }
  std::span<const std::size_t> ids() const { return ids_; }

  void push(std::size_t id, std::span<const std::uint64_t> l0);
  void reserve(std::size_t n);

 private:
  std::size_t l0_words_per_vec_;
  AlignedVector<std::uint64_t> codes_;
  std::vector<std::size_t> ids_;
};

/// Unique 8-bit parent keys in a contiguous byte row; each owns an L0 child group.
class ParentStore {
 public:
  static constexpr std::size_t kMaxParents = 1u << quantize::PARENT_BITS;
  static constexpr std::size_t kInvalidGroup = std::numeric_limits<std::size_t>::max();

  ParentStore(std::size_t l0_words_per_vec, std::size_t padded_dim);
  static ParentStore with_capacity(std::size_t l0_words_per_vec, std::size_t padded_dim,
                                   std::size_t vector_count);

  std::size_t l0_words_per_vec() const { return l0_words_per_vec_; }
  std::size_t padded_dim() const { return padded_dim_; }
  std::size_t unique_parent_count() const { return keys_.size(); }
  std::size_t total_vectors() const { return total_vectors_; }

  std::span<const std::uint8_t> unique_keys() const { return keys_; }
  const ParentGroup* group_for_key(std::uint8_t key) const;

  void push_vector(std::uint8_t parent_key, std::span<const std::uint64_t> l0, std::size_t id);
  void reserve_vectors(std::size_t vector_count);

 private:
  std::size_t l0_words_per_vec_;
  std::size_t padded_dim_;
  std::size_t total_vectors_ = 0;
  std::size_t reserve_hint_ = 0;
  AlignedVector<std::uint8_t> keys_;
  std::array<std::size_t, kMaxParents> by_key_{};
  std::vector<ParentGroup> groups_;
};

}  // namespace vectorcache::ingest
