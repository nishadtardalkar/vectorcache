#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorcache {

/// Orthonormal landmark matrix L (dim x m, column-major: L[d + j*dim]).
/// Built via QR of a seeded Gaussian matrix (thin Q's first m columns).
std::vector<float> make_ortho_landmarks(std::size_t dim, std::size_t m, std::uint64_t seed);

/// Int8 column-quantized landmark basis for query projection.
/// Codes are column-major with stride `dim_pad` (>= dim, multiple of 16):
///   codes[i + j * dim_pad], i < dim used; pad bytes are 0.
/// Reconstruct column j ≈ scales[j] * int8_column_j.
struct QuantizedLandmarks {
  std::vector<std::int8_t> codes;  // dim_pad * m
  std::vector<float> scales;       // m
  std::size_t dim = 0;
  std::size_t dim_pad = 0;
  std::size_t m = 0;
};

/// Per-column absmax → int8. `L` is float column-major dim*m.
QuantizedLandmarks quantize_landmarks(std::span<const float> L, std::size_t dim, std::size_t m);

/// q_land[qi, 0:m) ≈ scales[j] * q_scale[qi] * ⟨q_i8[qi], L_i8[:,j]⟩
/// Asymmetric: query quantized per-row to int8; L already int8.
void project_onto_landmarks_i8(const float* q_rot, std::size_t nq, std::size_t dim,
                               const QuantizedLandmarks& L, float* q_land, std::size_t scan_dim);

}  // namespace vectorcache
