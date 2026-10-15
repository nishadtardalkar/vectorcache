#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

struct Calibration {
  std::vector<float> shift;
  std::vector<float> scale_tq;
};

float row_norm(std::span<const float> row);

void encode(std::span<const float> vectors, std::size_t n, std::size_t dim, const Rotation& rotation,
            std::span<const float> boundaries, std::span<const float> centroids, std::size_t bit_width,
            const Calibration* calibration, std::vector<float>& rotated_scratch,
            std::vector<std::uint8_t>& packed_out, std::vector<float>& scales_out);

/// Quantize already-prepared rows (no rotation). `rotated` is n*dim; `norms` length n.
void encode_prerotated(std::span<const float> rotated, std::span<const float> norms, std::size_t n,
                       std::size_t dim, std::span<const float> boundaries,
                       std::span<const float> centroids, std::size_t bit_width,
                       const Calibration* calibration, std::vector<std::uint8_t>& packed_out,
                       std::vector<float>& scales_out);

Calibration fit_calibration(std::span<const float> vectors, std::size_t n, std::size_t dim,
                            const Rotation& rotation, std::span<const float> centroids,
                            std::vector<float>& rotated_scratch);

/// Beta CDF helper used by TQ+ anchor (exported for tests).
double beta_cdf_aa(double a, double t);

}  // namespace vectorcache
