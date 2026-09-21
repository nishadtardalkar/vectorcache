#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

inline constexpr std::size_t kMinBitsPerDim = 1;
inline constexpr std::size_t kMaxBitsPerDim = 8;

/// Bits per dimension (scalar TurboQuant).
struct QuantSpec {
  std::size_t bits = 1;
};

/// Validate bits-per-dim is in [kMinBitsPerDim, kMaxBitsPerDim].
void validate_bits_per_dim(std::size_t bits);

/// Validate bits and dim (>= 2).
void validate_quant_spec(std::size_t dim, QuantSpec spec);

/// Number of codebook codes for a vector of `dim` floats (= dim).
std::size_t num_blocks(std::size_t dim);

/// Total L0 bits for a vector of dim floats at bits-per-dim.
std::size_t l0_bits_per_vector(std::size_t dim, std::size_t bits = 1);

/// Number of u64 words needed to store L0 codes for dim floats at bits-per-dim.
std::size_t l0_words_per_vector(std::size_t dim, std::size_t bits = 1);

/// Lloyd-Max scalar codebook (TurboQuant / TurboVec): Beta((dim-1)/2,(dim-1)/2) on [-1,1].
class LloydMaxCodebook {
 public:
  LloydMaxCodebook() = default;
  LloydMaxCodebook(std::size_t dim, std::size_t bits);
  LloydMaxCodebook(std::size_t dim, QuantSpec spec);

  std::size_t srht_dim() const { return dim_; }
  std::size_t dim() const { return dim_; }
  std::size_t bits() const { return bits_; }
  std::size_t num_blocks() const { return dim_; }
  std::size_t num_centroids() const { return num_centroids_; }

  /// Flat centroids: K floats.
  std::span<const float> centroids() const { return centroids_; }

  /// Centroid k as a single-float span.
  std::span<const float> centroid(std::uint32_t index) const;

  /// 1D decision boundaries.
  std::span<const float> boundaries() const { return boundaries_; }

  /// Scalar encode.
  std::uint32_t encode(float x) const;

  /// Scalar centroid coordinate.
  float centroid_at(std::uint32_t index) const;

 private:
  std::size_t dim_ = 0;
  std::size_t bits_ = 0;
  std::size_t num_centroids_ = 0;
  std::vector<float> centroids_;   // K
  std::vector<float> boundaries_;  // size K-1
};

/// Quantize a rotated vector to packed n-bit codes (little-endian bitstream).
/// Returns total bits written (dim * bits).
std::size_t quantize_blocks_to_nbit_into(std::span<const float> vector,
                                         const LloydMaxCodebook& codebook,
                                         std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_blocks_to_nbit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_blocks_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook);

/// Alias for scalar path (tests / call sites).
std::size_t quantize_1dim_to_nbit_into(std::span<const float> vector, const LloydMaxCodebook& codebook,
                                       std::span<std::uint64_t> out);

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook);

/// Unpack centroid index for dim `block_index` from a packed L0 code row.
std::uint32_t unpack_code(std::span<const std::uint64_t> words, std::size_t block_index,
                          std::size_t bits);

/// RaBitQ / TurboVec length-renorm scale: ||u|| / <u, x_hat> for unit u (||u||=1 → 1/<u,x_hat>).
float ip_scale_alpha(std::span<const float> rotated_unit, std::span<const std::uint64_t> codes,
                     const LloydMaxCodebook& codebook);

}  // namespace vectorcache::quantize
