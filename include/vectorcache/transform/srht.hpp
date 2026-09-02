#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorcache::transform {

/// TurboQuant-style 3-round SRHT: H·D₃·H·D₂·H·D₁ with orthonormal H.
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
  std::vector<std::int8_t> signs1_;
  std::vector<std::int8_t> signs2_;
  std::vector<std::int8_t> signs3_;
  mutable std::vector<float> fwht_scratch_;
};

}  // namespace vectorcache::transform
