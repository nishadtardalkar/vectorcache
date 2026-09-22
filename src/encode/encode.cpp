#include "vectorcache/encode/encode.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>

#include "vectorcache/constants.hpp"

namespace vectorcache {
namespace {

float combine_norm_chains(const float c[kNormChains]) {
  return ((c[0] + c[1]) + (c[2] + c[3])) + ((c[4] + c[5]) + (c[6] + c[7]));
}

float norm_sq_scalar(std::span<const float> row) {
  float chains[kNormChains] = {};
  for (std::size_t j = 0; j < row.size(); ++j) {
    chains[j % kNormChains] += row[j] * row[j];
  }
  return combine_norm_chains(chains);
}

double recon_entry(float centroid, double inv, double sh) {
  return static_cast<double>(centroid) * inv - sh;
}

float scale_from_inner(double inner, float norm) {
  if (inner > kDegenerateInnerEps) {
    return norm / static_cast<float>(inner);
  }
  return 0.f;
}

// Incomplete beta / Beta(a,a) — duplicated lightly from codebook for TQ+ anchor.
double log_gamma_stirling(double z) {
  static constexpr double c[9] = {0.99999999999980993, 676.5203681218851,   -1259.1392167224028,
                                  771.32342877765313,  -176.61502916214059, 12.507343278686905,
                                  -0.13857109526572012, 9.984369654078861e-6, 1.5056327351493116e-7};
  if (z < 0.5) {
    return std::log(3.14159265358979323846) - std::log(std::sin(3.14159265358979323846 * z)) -
           log_gamma_stirling(1.0 - z);
  }
  z -= 1.0;
  double x = c[0];
  for (int i = 1; i < 9; ++i) {
    x += c[i] / (z + static_cast<double>(i));
  }
  const double t = z + 7.5;
  return 0.5 * std::log(2.0 * 3.14159265358979323846) + (z + 0.5) * std::log(t) - t + std::log(x);
}

double betacf(double a, double b, double x) {
  constexpr int kMaxIt = 200;
  constexpr double kEps = 3e-14;
  constexpr double kFpmin = 1e-30;
  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpmin) d = kFpmin;
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= kMaxIt; ++m) {
    const int m2 = 2 * m;
    double aa = static_cast<double>(m) * (b - m) * x / ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) d = kFpmin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) c = kFpmin;
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) d = kFpmin;
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) c = kFpmin;
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) break;
  }
  return h;
}

double incomplete_beta(double a, double b, double x) {
  if (x <= 0.0) return 0.0;
  if (x >= 1.0) return 1.0;
  const double lbeta = log_gamma_stirling(a) + log_gamma_stirling(b) - log_gamma_stirling(a + b);
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / a * betacf(a, b, x);
  }
  return 1.0 -
         std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / b * betacf(b, a, 1.0 - x);
}

std::uint32_t f32_sort_key(float x) {
  const std::uint32_t b = std::bit_cast<std::uint32_t>(x);
  return b ^ (static_cast<std::uint32_t>(static_cast<std::int32_t>(b) >> 31) | 0x80000000u);
}

float f32_from_sort_key(std::uint32_t k) {
  const std::uint32_t mask = (k & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
  return std::bit_cast<float>(k ^ mask);
}

float fused_quantize_scale_pack(std::span<const float> rot_orig, std::span<const float> shift,
                                std::span<const float> scale_tq, std::span<const float> inv_scale_tq,
                                std::span<const float> boundaries, std::span<const float> centroids,
                                float norm, std::span<std::uint8_t> packed_row, std::size_t dim,
                                std::size_t bits, std::size_t bytes_per_plane) {
  double chains[4] = {};
  const std::size_t chunks = dim / 8;
  const std::size_t n_bounds = (1u << bits) - 1;
  for (std::size_t c = 0; c < chunks; ++c) {
    const std::size_t offset = c * 8;
    std::uint8_t codes[8];
    for (std::size_t k = 0; k < 8; ++k) {
      const std::size_t j = offset + k;
      const float calib = (rot_orig[j] + shift[j]) * scale_tq[j];
      std::uint8_t v = 0;
      for (std::size_t bi = 0; bi < n_bounds; ++bi) {
        if (calib > boundaries[bi]) {
          ++v;
        }
      }
      codes[k] = v;
      const double centroid_in_orig =
          recon_entry(centroids[v], static_cast<double>(inv_scale_tq[j]),
                      static_cast<double>(shift[j]));
      chains[j % 4] += static_cast<double>(rot_orig[j]) * centroid_in_orig;
    }
    for (std::size_t p = 0; p < bits; ++p) {
      std::uint8_t byte = 0;
      for (std::size_t k = 0; k < 8; ++k) {
        byte |= static_cast<std::uint8_t>(((codes[k] >> p) & 1) << (7 - k));
      }
      packed_row[p * bytes_per_plane + c] = byte;
    }
  }
  const double inner = (chains[0] + chains[1]) + (chains[2] + chains[3]);
  return scale_from_inner(inner, norm);
}

std::vector<float> rotate_batch_into(std::span<const float> vectors, std::size_t n, std::size_t dim,
                                     const Rotation& rotation, std::vector<float>& rotated_scratch) {
  std::vector<float> norms(n);
  rotated_scratch.assign(n * dim, 0.f);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> scratch(dim);
    std::vector<float> row(dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(n); ++i) {
      auto src = vectors.subspan(static_cast<std::size_t>(i) * dim, dim);
      const float n_val = row_norm(src);
      norms[static_cast<std::size_t>(i)] = n_val;
      const float inv = (n_val > kMinInputNorm) ? (1.f / n_val) : 0.f;
      rotation.apply_scaled_into(src, inv, row, scratch);
      std::copy(row.begin(), row.end(),
                rotated_scratch.begin() + static_cast<std::ptrdiff_t>(i) * static_cast<std::ptrdiff_t>(dim));
    }
  }
  return norms;
}

void encode_prerotated(std::span<const float> rotated, std::span<const float> norms, std::size_t n,
                       std::size_t dim, std::span<const float> boundaries,
                       std::span<const float> centroids, std::size_t bit_width,
                       const Calibration* calibration, std::vector<std::uint8_t>& packed_out,
                       std::vector<float>& scales_out) {
  std::vector<float> identity_shift;
  std::vector<float> identity_scale;
  std::span<const float> shift;
  std::span<const float> scale_tq;
  if (calibration) {
    shift = calibration->shift;
    scale_tq = calibration->scale_tq;
  } else {
    identity_shift.assign(dim, 0.f);
    identity_scale.assign(dim, 1.f);
    shift = identity_shift;
    scale_tq = identity_scale;
  }
  std::vector<float> inv_scale_tq(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    inv_scale_tq[d] = 1.f / scale_tq[d];
  }

  const std::size_t bytes_per_plane = dim / 8;
  const std::size_t bytes_per_row = bit_width * bytes_per_plane;
  const std::size_t packed_old = packed_out.size();
  const std::size_t scales_old = scales_out.size();
  packed_out.resize(packed_old + n * bytes_per_row);
  scales_out.resize(scales_old + n);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < static_cast<int>(n); ++i) {
    auto rot = rotated.subspan(static_cast<std::size_t>(i) * dim, dim);
    auto packed_row =
        std::span<std::uint8_t>(packed_out.data() + packed_old + static_cast<std::size_t>(i) * bytes_per_row,
                                bytes_per_row);
    scales_out[scales_old + static_cast<std::size_t>(i)] = fused_quantize_scale_pack(
        rot, shift, scale_tq, inv_scale_tq, boundaries, centroids, norms[static_cast<std::size_t>(i)],
        packed_row, dim, bit_width, bytes_per_plane);
  }
}

std::pair<double, double> tqplus_anchor_probs(double a, std::span<const float> centroids) {
  float c_outer = 0.f;
  for (float c : centroids) {
    c_outer = std::max(c_outer, std::fabs(c));
  }
  const double p_hi = incomplete_beta(a, a, (static_cast<double>(c_outer) + 1.0) / 2.0);
  return {1.0 - p_hi, p_hi};
}

Calibration compute_tqplus_calibration(std::span<const float> rotated, std::size_t n, std::size_t dim,
                                       std::span<const float> centroids) {
  Calibration out;
  out.shift.assign(dim, 0.f);
  out.scale_tq.assign(dim, 1.f);
  if (n < kMinCalibrationRows) {
    return out;
  }
  const double a = (static_cast<double>(dim) - 1.0) / 2.0;
  float c_outer = 0.f;
  for (float c : centroids) {
    c_outer = std::max(c_outer, std::fabs(c));
  }
  const auto [p_lo, p_hi] = tqplus_anchor_probs(a, centroids);
  const float qc_lo = -c_outer;
  const float qc_hi = c_outer;
  const float qc_span = qc_hi - qc_lo;

  std::size_t lo_idx = static_cast<std::size_t>(static_cast<double>(n) * p_lo);
  std::size_t hi_idx = static_cast<std::size_t>(static_cast<double>(n) * p_hi);
  hi_idx = std::min(hi_idx, n - 1);
  hi_idx = std::max(hi_idx, lo_idx + 1);

  std::vector<std::uint32_t> col(n);
  for (std::size_t d = 0; d < dim; ++d) {
    for (std::size_t i = 0; i < n; ++i) {
      col[i] = f32_sort_key(rotated[i * dim + d]);
    }
    // nth_element for lo and hi order statistics
    std::nth_element(col.begin(), col.begin() + static_cast<std::ptrdiff_t>(lo_idx), col.end());
    const float q_lo = f32_from_sort_key(col[lo_idx]);
    std::nth_element(col.begin() + static_cast<std::ptrdiff_t>(lo_idx) + 1,
                     col.begin() + static_cast<std::ptrdiff_t>(hi_idx), col.end());
    const float q_hi = f32_from_sort_key(col[hi_idx]);
    const float span = q_hi - q_lo;
    if (!(span > 1e-12f) || !(qc_span > 0.f)) {
      continue;
    }
    const float scale = qc_span / span;
    const float shift = qc_lo / scale - q_lo;
    out.shift[d] = shift;
    out.scale_tq[d] = scale;
  }
  return out;
}

}  // namespace

float row_norm(std::span<const float> row) { return std::sqrt(norm_sq_scalar(row)); }

double beta_cdf_aa(double a, double t) { return incomplete_beta(a, a, t); }

void encode(std::span<const float> vectors, std::size_t n, std::size_t dim, const Rotation& rotation,
            std::span<const float> boundaries, std::span<const float> centroids, std::size_t bit_width,
            const Calibration* calibration, std::vector<float>& rotated_scratch,
            std::vector<std::uint8_t>& packed_out, std::vector<float>& scales_out) {
  if (dim == 0 || dim % 8 != 0) {
    throw std::invalid_argument("encode requires dim multiple of 8");
  }
  if (bit_width < 2 || bit_width > 4) {
    throw std::invalid_argument("bit_width must be 2, 3, or 4");
  }
  auto norms = rotate_batch_into(vectors, n, dim, rotation, rotated_scratch);
  encode_prerotated(rotated_scratch, norms, n, dim, boundaries, centroids, bit_width, calibration,
                    packed_out, scales_out);
}

Calibration fit_calibration(std::span<const float> vectors, std::size_t n, std::size_t dim,
                            const Rotation& rotation, std::span<const float> centroids,
                            std::vector<float>& rotated_scratch) {
  if (n < kMinCalibrationRows) {
    throw std::invalid_argument("calibration needs at least 2 rows");
  }
  rotate_batch_into(vectors, n, dim, rotation, rotated_scratch);
  return compute_tqplus_calibration(rotated_scratch, n, dim, centroids);
}

}  // namespace vectorcache
