#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/transform/srht_config.hpp"

namespace vectorcache::transform {

/// TurboQuant-style SRHT: H·Dₙ·…·H·D₁ with orthonormal H (round count set at compile time).
class SrhtRotation {
 public:
  SrhtRotation(std::size_t original_dim, std::uint64_t seed);

  std::size_t original_dim() const { return original_dim_; }
  std::size_t padded_dim() const { return padded_dim_; }

  /// Zero-pad vector, apply SRHT, write result into out (length padded_dim).
  void apply(std::span<const float> vector, std::span<float> out) const;

  /// Apply SRHT in-place on a padded, normalized buffer (length padded_dim).
  void apply_in_place(std::span<float> buf) const;

 private:
  void apply_rounds(std::span<float> buf) const;

  std::size_t original_dim_;
  std::size_t padded_dim_;
  float inv_sqrt_n_;
#if VECTORCACHE_SRHT_ROUNDS >= 1
  AlignedVector<float> signs1_f_;
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 2
  AlignedVector<float> signs2_f_;
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 3
  AlignedVector<float> signs3_f_;
#endif
  mutable AlignedVector<float> fwht_scratch_;
};

}  // namespace vectorcache::transform
