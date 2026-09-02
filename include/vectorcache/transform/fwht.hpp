#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace vectorcache::transform {

/// Next power of two >= dim (minimum 1 for dim > 0).
std::size_t padded_dim(std::size_t dim);

/// In-place unnormalized Walsh-Hadamard butterfly transform.
/// buf.size() must be a power of two.
void fwht_in_place(std::span<float> buf);

/// Apply orthonormal FWHT: unnormalized butterfly followed by 1/sqrt(N) scaling.
void fwht_orthonormal_in_place(std::span<float> buf);

/// Orthonormal FWHT with precomputed 1/sqrt(N) scale folded into the last stage.
void fwht_orthonormal_in_place(std::span<float> buf, float inv_sqrt_n);

/// Stockham (ping-pong) orthonormal FWHT; scratch must match buf.size().
void fwht_stockham_orthonormal_in_place(std::span<float> buf, std::span<float> scratch,
                                        float inv_sqrt_n);

/// Multiply all elements by scale using AVX-512.
void scale_in_place(std::span<float> buf, float scale);

/// Element-wise buf[i] *= signs[i] where signs[i] is +1 or -1.
void apply_signs_i8(std::span<float> buf, std::span<const std::int8_t> signs);

}  // namespace vectorcache::transform
