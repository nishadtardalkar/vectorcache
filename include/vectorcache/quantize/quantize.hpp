#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

inline constexpr std::size_t kMinBitsPerDim = 1;
inline constexpr std::size_t kMaxBitsPerDim = 8;

/// Validate bits-per-dim is in [kMinBitsPerDim, kMaxBitsPerDim].
void validate_bits_per_dim(std::size_t bits);

/// Total L0 bits for a vector of dim floats at bits-per-dim.
std::size_t l0_bits_per_vector(std::size_t dim, std::size_t bits = 1);

/// Number of u64 words needed to store L0 codes for dim floats at bits-per-dim.
std::size_t l0_words_per_vector(std::size_t dim, std::size_t bits = 1);

/// Lloyd-Max codebook for N(0, 1/srht_dim) with 2^bits centroids (TurboQuantMSE).
class LloydMaxCodebook {
 public:
  LloydMaxCodebook() = default;
  LloydMaxCodebook(std::size_t srht_dim, std::size_t bits);

  std::size_t srht_dim() const { return srht_dim_; }
  std::size_t bits() const { return bits_; }
  std::size_t num_centroids() const { return centroids_.size(); }
  std::span<const float> centroids() const { return centroids_; }
  std::span<const float> boundaries() const { return boundaries_; }

  /// Nearest-centroid index in [0, 2^bits).
  std::uint32_t encode(float x) const;

  float centroid_at(std::uint32_t index) const { return centroids_[index]; }

 private:
  std::size_t srht_dim_ = 0;
  std::size_t bits_ = 0;
  std::vector<float> centroids_;
  std::vector<float> boundaries_;  // size K-1; region i is (-inf, b0], (b0,b1], ...
};

/// Quantize a rotated vector to packed n-bit centroid indices (little-endian bitstream).
/// Returns total bits written (dim * bits).
std::size_t quantize_1dim_to_nbit_into(std::span<const float> vector, const LloydMaxCodebook& codebook,
                                       std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_1dim_to_nbit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook);

/// Unpack centroid index for dimension `dim_index` from a packed L0 code row.
std::uint32_t unpack_code(std::span<const std::uint64_t> words, std::size_t dim_index,
                          std::size_t bits);

}  // namespace vectorcache::quantize
