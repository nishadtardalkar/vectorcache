#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorcache {

/// Orthonormal landmark matrix L (dim x m, column-major: L[d + j*dim]).
/// Built via QR of a seeded Gaussian matrix (thin Q's first m columns).
std::vector<float> make_ortho_landmarks(std::size_t dim, std::size_t m, std::uint64_t seed);

/// 4-bit column-quantized landmark basis for query projection.
/// Packed codes are column-major with `bytes_per_col = dim_pad/2` (`dim_pad` multiple of 16):
///   each byte holds two signed levels in 0..14 as (code+7), low nibble = even dim.
/// Reconstruct column j ≈ scales[j] * unpacked_i8_column_j  (levels in [-7, 7]).
struct QuantizedLandmarks {
  std::vector<std::uint8_t> codes;  // (dim_pad/2) * m
  std::vector<float> scales;        // m
  std::size_t dim = 0;
  std::size_t dim_pad = 0;
  std::size_t m = 0;
};

/// Per-column absmax → signed 4-bit. `L` is float column-major dim*m.
QuantizedLandmarks quantize_landmarks(std::span<const float> L, std::size_t dim, std::size_t m);

/// q_land[qi, 0:m) ≈ scales[j] * q_scale[qi] * ⟨q_i8[qi], unpack4(L[:,j])⟩
/// Asymmetric: query int8 per row; L stored 4-bit packed.
void project_onto_landmarks_q4(const float* q_rot, std::size_t nq, std::size_t dim,
                               const QuantizedLandmarks& L, float* q_land, std::size_t scan_dim);

}  // namespace vectorcache
