#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

inline constexpr std::size_t kMinBitsPerDim = 1;
inline constexpr std::size_t kMaxBitsPerDim = 8;
inline constexpr std::size_t kMinBlockDims = 1;
inline constexpr std::size_t kMaxBlockDims = 16;

/// Bits per codebook block (historically "per dim" when block_dims == 1).
struct QuantSpec {
  std::size_t block_dims = 1;
  std::size_t bits = 1;
};

/// Validate bits-per-block is in [kMinBitsPerDim, kMaxBitsPerDim].
void validate_bits_per_dim(std::size_t bits);

/// Validate block_dims is in [kMinBlockDims, kMaxBlockDims].
void validate_block_dims(std::size_t block_dims);

/// Validate dim is divisible by block_dims and both are in range.
void validate_quant_spec(std::size_t dim, QuantSpec spec);

/// Number of codebook blocks for a vector of `dim` floats.
std::size_t num_blocks(std::size_t dim, std::size_t block_dims = 1);

/// Total L0 bits for a vector of dim floats at bits-per-block.
std::size_t l0_bits_per_vector(std::size_t dim, std::size_t bits = 1, std::size_t block_dims = 1);

/// Number of u64 words needed to store L0 codes for dim floats at bits-per-block.
std::size_t l0_words_per_vector(std::size_t dim, std::size_t bits = 1, std::size_t block_dims = 1);

/// Lloyd-Max / block-VQ codebook (TurboQuant / TurboVec).
/// - block_dims == 1: scalar Beta((dim-1)/2,(dim-1)/2) on [-1,1], 2^bits centroids.
/// - block_dims  > 1: product-Beta VQ in R^{block_dims} with 2^bits vector centroids.
class LloydMaxCodebook {
 public:
  LloydMaxCodebook() = default;
  LloydMaxCodebook(std::size_t dim, std::size_t bits, std::size_t block_dims = 1);
  LloydMaxCodebook(std::size_t dim, QuantSpec spec);

  std::size_t srht_dim() const { return dim_; }
  std::size_t dim() const { return dim_; }
  std::size_t bits() const { return bits_; }
  std::size_t block_dims() const { return block_dims_; }
  std::size_t num_blocks() const { return dim_ / block_dims_; }
  std::size_t num_centroids() const { return num_centroids_; }

  /// Flat row-major centroids: K * block_dims floats.
  std::span<const float> centroids() const { return centroids_; }

  /// Centroid k as a span of length block_dims.
  std::span<const float> centroid(std::uint32_t index) const;

  /// 1D decision boundaries (only populated when block_dims == 1).
  std::span<const float> boundaries() const { return boundaries_; }

  /// Scalar encode (block_dims == 1 only).
  std::uint32_t encode(float x) const;

  /// Block encode: nearest centroid in R^{block_dims}.
  std::uint32_t encode(std::span<const float> block) const;

  /// Scalar centroid coordinate (block_dims == 1), or first coord of block centroid.
  float centroid_at(std::uint32_t index) const;

 private:
  std::size_t dim_ = 0;
  std::size_t bits_ = 0;
  std::size_t block_dims_ = 1;
  std::size_t num_centroids_ = 0;
  std::vector<float> centroids_;   // K * block_dims
  std::vector<float> boundaries_;  // size K-1 when block_dims == 1
};

/// Quantize a rotated vector to packed n-bit block codes (little-endian bitstream).
/// Returns total bits written ((dim/block_dims) * bits).
std::size_t quantize_blocks_to_nbit_into(std::span<const float> vector,
                                         const LloydMaxCodebook& codebook,
                                         std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_blocks_to_nbit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_blocks_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook);

/// Alias for block_dims == 1 path (tests / call sites).
std::size_t quantize_1dim_to_nbit_into(std::span<const float> vector, const LloydMaxCodebook& codebook,
                                       std::span<std::uint64_t> out);

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook);

/// Unpack centroid index for block `block_index` from a packed L0 code row.
std::uint32_t unpack_code(std::span<const std::uint64_t> words, std::size_t block_index,
                          std::size_t bits);

/// RaBitQ / TurboVec length-renorm scale: ||u|| / <u, x_hat> for unit u (||u||=1 → 1/<u,x_hat>).
float ip_scale_alpha(std::span<const float> rotated_unit, std::span<const std::uint64_t> codes,
                     const LloydMaxCodebook& codebook);

}  // namespace vectorcache::quantize
