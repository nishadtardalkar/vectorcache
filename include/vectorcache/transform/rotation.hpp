#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"

namespace vectorcache {

/// Deterministic orthogonal FWHT rotation (K=`kRotationRounds` rounds of
/// Fisher–Yates perm + ±1 signs + normalized block Walsh–Hadamard).
class Rotation {
 public:
  explicit Rotation(std::size_t dim);

  std::size_t dim() const { return dim_; }
  std::size_t block() const { return block_; }

  void apply(std::span<float> row) const;
  void apply_with_scratch(std::span<float> row, std::span<float> scratch) const;
  void apply_scaled_into(std::span<const float> src, float inv, std::span<float> dst,
                         std::span<float> scratch) const;

 private:
  std::size_t dim_ = 0;
  std::size_t block_ = 0;
  float inv_sqrt_block_ = 0.f;
  std::vector<std::vector<float>> signs_;
  std::vector<std::vector<std::uint32_t>> perms_;
  std::vector<float> signs1_pre_;  // unused for K=1; kept for ABI stability of layout
};

}  // namespace vectorcache
