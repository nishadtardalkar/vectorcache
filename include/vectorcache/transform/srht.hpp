#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/transform/srht_config.hpp"

namespace vectorcache::transform {

/// TurboVec-style orthogonal rotation: K rounds of
/// (global Fisher–Yates perm → ±1 signs → normalized block Walsh–Hadamard).
/// Working length equals `original_dim` (no zero-pad). Block size is the largest
/// power-of-two divisor of dim ( ≥1 ). Round count is compile-time (`VECTORCACHE_SRHT_ROUNDS`).
class SrhtRotation {
 public:
  SrhtRotation(std::size_t original_dim, std::uint64_t seed);

  /// Rotation length (= original_dim). Alias kept for call-site compatibility.
  std::size_t srht_dim() const { return dim_; }
  std::size_t padded_dim() const { return dim_; }
  std::size_t block_size() const { return block_; }

  /// Copy vector and apply rotation into out (both length dim).
  void apply(std::span<const float> vector, std::span<float> out) const;

  /// Apply rotation in-place on a buffer of length dim.
  void apply_in_place(std::span<float> buf) const;

 private:
  void apply_rounds(std::span<float> buf) const;

  std::size_t dim_;
  std::size_t block_;
  float inv_sqrt_block_;
  // Per-round signs (+1/-1) and permutations (out[i] = in[perm[i]] * sign[i]).
  std::vector<AlignedVector<float>> signs_;
  std::vector<std::vector<std::uint32_t>> perms_;
};

}  // namespace vectorcache::transform
