#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"

namespace vectorcache::ingest {

/// Flat in-RAM index: contiguous L0 codes + parallel ids.
class VectorStore {
 public:
  VectorStore(std::size_t l0_words_per_vec, std::size_t input_dim, std::size_t srht_dim,
              std::size_t bits_per_dim = 1);
  static VectorStore with_capacity(std::size_t l0_words_per_vec, std::size_t input_dim,
                                   std::size_t srht_dim, std::size_t vector_count,
                                   std::size_t bits_per_dim = 1);

  std::size_t l0_words_per_vec() const { return l0_words_per_vec_; }
  std::size_t bits_per_dim() const { return bits_per_dim_; }
  std::size_t input_dim() const { return input_dim_; }
  std::size_t srht_dim() const { return srht_dim_; }
  std::size_t size() const { return ids_.size(); }
  bool empty() const { return ids_.empty(); }

  std::span<const std::uint64_t> vector_l0(std::size_t index) const;
  std::span<const std::uint64_t> l0_codes() const { return codes_; }
  std::span<const std::size_t> ids() const { return ids_; }
  std::size_t id_at(std::size_t index) const;

  void push(std::size_t id, std::span<const std::uint64_t> l0);
  void reserve(std::size_t n);

 private:
  std::size_t l0_words_per_vec_;
  std::size_t bits_per_dim_;
  std::size_t input_dim_;
  std::size_t srht_dim_;
  AlignedVector<std::uint64_t> codes_;
  std::vector<std::size_t> ids_;
};

}  // namespace vectorcache::ingest
