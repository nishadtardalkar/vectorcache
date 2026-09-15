#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/quantize/dim_postings.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::ingest {

/// Contiguous L0 codes + explicit ids for one support key.
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

/// Open-addressing map SupportKey → ParentGroup (single posting per vector).
class ParentStore {
 public:
  static constexpr std::size_t kInvalidGroup = std::numeric_limits<std::size_t>::max();

  ParentStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
              std::size_t top_d);
  static ParentStore with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                   std::size_t srht_dim, std::size_t top_d,
                                   std::size_t vector_count);

  std::size_t l0_words_per_vec() const { return l0_words_per_vec_; }
  std::size_t input_dim() const { return input_dim_; }
  std::size_t srht_dim() const { return srht_dim_; }
  std::size_t top_d() const { return top_d_; }
  std::size_t unique_parent_count() const { return groups_.size(); }
  std::size_t total_vectors() const { return total_vectors_; }

  std::span<const quantize::SupportKey> unique_keys() const { return keys_; }
  const ParentGroup* find(const quantize::SupportKey& key) const;
  /// Direct group lookup by key index (parallel to unique_keys()).
  const ParentGroup& group_at(std::size_t key_idx) const;
  const quantize::DimPostingIndex& dim_postings() const;

  /// Post L0 under one support key; increments total_vectors by one.
  void push_vector(const quantize::SupportKey& key, std::span<const std::uint64_t> l0,
                   std::size_t id);

  void reserve_vectors(std::size_t vector_count);

 private:
  struct Slot {
    quantize::SupportKey key{};
    /// kInvalidGroup means empty slot (no separate occupied flag).
    std::size_t group_idx = kInvalidGroup;
  };

  void rehash(std::size_t new_cap);
  std::size_t probe_slot(const quantize::SupportKey& key) const;
  void insert_mapping(const quantize::SupportKey& key, std::size_t group_idx);
  void invalidate_dim_postings() const;
  static bool slot_occupied(const Slot& s) { return s.group_idx != kInvalidGroup; }

  std::size_t l0_words_per_vec_;
  std::size_t input_dim_;
  std::size_t srht_dim_;
  std::size_t top_d_;
  std::size_t total_vectors_ = 0;
  std::size_t map_size_ = 0;
  std::vector<Slot> slots_;
  std::vector<quantize::SupportKey> keys_;
  std::vector<ParentGroup> groups_;
  mutable std::optional<quantize::DimPostingIndex> dim_postings_;
  mutable std::size_t dim_postings_key_count_ = 0;
};

}  // namespace vectorcache::ingest
