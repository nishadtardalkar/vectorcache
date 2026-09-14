#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/aligned.hpp"
#include "vectorcache/transform/srht_config.hpp"

namespace vectorcache::transform {

/// TurboQuant-style SRHT: H·Dₙ·…·H·D₁ with orthonormal H (round count set at compile time).
/// Zero-padding to next power of two happens only inside apply() / caller scratch for FWHT.
class SrhtRotation {
 public:
  SrhtRotation(std::size_t original_dim, std::uint64_t seed);

  /// FWHT length (= next power of two). Prefer this name; padded_dim() is an alias.
  std::size_t srht_dim() const { return srht_dim_; }
  std::size_t padded_dim() const { return srht_dim_; }

  /// Copy vector, zero-pad [original_dim, srht_dim), apply SRHT into out (length srht_dim).
  void apply(std::span<const float> vector, std::span<float> out) const;

  /// Apply SRHT in-place on a buffer of length srht_dim (pad already applied by caller).
  void apply_in_place(std::span<float> buf) const;

 private:
  void apply_rounds(std::span<float> buf) const;
  AlignedVector<float>& fwht_scratch() const;

  std::size_t original_dim_;
  std::size_t srht_dim_;
  float inv_sqrt_n_;
#if VECTORCACHE_SRHT_ROUNDS >= 1
  AlignedVector<std::int8_t> signs1_;
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 2
  AlignedVector<std::int8_t> signs2_;
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 3
  AlignedVector<std::int8_t> signs3_;
#endif
};

}  // namespace vectorcache::transform
