#include "vectorcache/transform/landmarks.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

inline std::size_t pad16(std::size_t n) { return (n + 15u) & ~std::size_t{15u}; }

inline std::int8_t quantize_to_i8(float x, float inv_scale) {
  const float q = std::round(x * inv_scale);
  const float clamped = std::max(-127.f, std::min(127.f, q));
  return static_cast<std::int8_t>(clamped);
}

#if defined(__x86_64__) || defined(_M_X64)
inline std::int32_t hsum256_epi32(__m256i v) {
  __m128i lo = _mm256_castsi256_si128(v);
  __m128i hi = _mm256_extracti128_si256(v, 1);
  lo = _mm_add_epi32(lo, hi);
  lo = _mm_hadd_epi32(lo, lo);
  lo = _mm_hadd_epi32(lo, lo);
  return _mm_cvtsi128_si32(lo);
}

/// Dot int8 vectors of length dim_pad (multiple of 16). Pad bytes must be 0.
inline std::int32_t dot_i8_avx2(const std::int8_t* a, const std::int8_t* b, std::size_t dim_pad) {
  __m256i acc = _mm256_setzero_si256();
  for (std::size_t i = 0; i < dim_pad; i += 16) {
    const __m128i a16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m128i b16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
    const __m256i aw = _mm256_cvtepi8_epi16(a16);
    const __m256i bw = _mm256_cvtepi8_epi16(b16);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(aw, bw));
  }
  return hsum256_epi32(acc);
}
#endif

inline std::int32_t dot_i8_scalar(const std::int8_t* a, const std::int8_t* b, std::size_t dim) {
  std::int32_t acc = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    acc += static_cast<std::int32_t>(a[i]) * static_cast<std::int32_t>(b[i]);
  }
  return acc;
}

}  // namespace

std::vector<float> make_ortho_landmarks(std::size_t dim, std::size_t m, std::uint64_t seed) {
  if (dim == 0 || m == 0 || m > dim) {
    throw std::invalid_argument("landmarks require 0 < m <= dim");
  }
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> A(static_cast<std::size_t>(dim) * m);
  for (double& v : A) {
    v = nd(rng);
  }

  // Modified Gram-Schmidt on columns (column-major).
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      double dot = 0.0;
      for (std::size_t r = 0; r < dim; ++r) {
        dot += A[r + i * dim] * A[r + j * dim];
      }
      for (std::size_t r = 0; r < dim; ++r) {
        A[r + i * dim] -= dot * A[r + j * dim];
      }
    }
    double nrm = 0.0;
    for (std::size_t r = 0; r < dim; ++r) {
      const double v = A[r + i * dim];
      nrm += v * v;
    }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-15) {
      throw std::runtime_error("degenerate landmark column");
    }
    const double inv = 1.0 / nrm;
    for (std::size_t r = 0; r < dim; ++r) {
      A[r + i * dim] *= inv;
    }
  }

  std::vector<float> L(static_cast<std::size_t>(dim) * m);
  for (std::size_t i = 0; i < A.size(); ++i) {
    L[i] = static_cast<float>(A[i]);
  }
  return L;
}

QuantizedLandmarks quantize_landmarks(std::span<const float> L, std::size_t dim, std::size_t m) {
  if (dim == 0 || m == 0 || L.size() < dim * m) {
    throw std::invalid_argument("quantize_landmarks: bad dim/m/L");
  }
  QuantizedLandmarks out;
  out.dim = dim;
  out.dim_pad = pad16(dim);
  out.m = m;
  out.scales.assign(m, 0.f);
  out.codes.assign(out.dim_pad * m, 0);

  for (std::size_t j = 0; j < m; ++j) {
    const float* col = L.data() + j * dim;
    float amax = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
      amax = std::max(amax, std::fabs(col[i]));
    }
    if (amax < 1e-30f) {
      out.scales[j] = 0.f;
      continue;
    }
    out.scales[j] = amax / 127.f;
    const float inv = 127.f / amax;
    std::int8_t* dest = out.codes.data() + j * out.dim_pad;
    for (std::size_t i = 0; i < dim; ++i) {
      dest[i] = quantize_to_i8(col[i], inv);
    }
  }
  return out;
}

void project_onto_landmarks_i8(const float* q_rot, std::size_t nq, std::size_t dim,
                               const QuantizedLandmarks& L, float* q_land, std::size_t scan_dim) {
  if (L.dim != dim || L.m == 0 || scan_dim < L.m) {
    throw std::invalid_argument("project_onto_landmarks_i8: shape mismatch");
  }
  const std::size_t m = L.m;
  const std::size_t dim_pad = L.dim_pad;
  const std::int8_t* codes = L.codes.data();
  const float* scales = L.scales.data();

  // Quantize each query row to int8 once, then int8×int8 dots vs landmark columns.
  std::vector<std::int8_t> q_i8(nq * dim_pad, 0);
  std::vector<float> q_scales(nq, 0.f);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    const std::size_t qis = static_cast<std::size_t>(qi);
    const float* q = q_rot + qis * dim;
    float amax = 0.f;
    for (std::size_t i = 0; i < dim; ++i) {
      amax = std::max(amax, std::fabs(q[i]));
    }
    std::int8_t* dest = q_i8.data() + qis * dim_pad;
    if (amax < 1e-30f) {
      q_scales[qis] = 0.f;
      continue;
    }
    q_scales[qis] = amax / 127.f;
    const float inv = 127.f / amax;
    for (std::size_t i = 0; i < dim; ++i) {
      dest[i] = quantize_to_i8(q[i], inv);
    }
  }

  constexpr std::size_t kLmTile = 8;
  const std::size_t m_aligned = m & ~(kLmTile - 1);

  for (std::size_t j0 = 0; j0 < m_aligned; j0 += kLmTile) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::int8_t* qrow = q_i8.data() + qis * dim_pad;
      const float qs = q_scales[qis];
      float* dest = q_land + qis * scan_dim + j0;
      for (std::size_t t = 0; t < kLmTile; ++t) {
        const std::int8_t* Lj = codes + (j0 + t) * dim_pad;
#if defined(__x86_64__) || defined(_M_X64)
        const std::int32_t dot = dot_i8_avx2(qrow, Lj, dim_pad);
#else
        const std::int32_t dot = dot_i8_scalar(qrow, Lj, dim);
#endif
        dest[t] = qs * scales[j0 + t] * static_cast<float>(dot);
      }
    }
  }

  for (std::size_t j = m_aligned; j < m; ++j) {
    const std::int8_t* Lj = codes + j * dim_pad;
    const float ls = scales[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::int8_t* qrow = q_i8.data() + qis * dim_pad;
#if defined(__x86_64__) || defined(_M_X64)
      const std::int32_t dot = dot_i8_avx2(qrow, Lj, dim_pad);
#else
      const std::int32_t dot = dot_i8_scalar(qrow, Lj, dim);
#endif
      q_land[qis * scan_dim + j] = q_scales[qis] * ls * static_cast<float>(dot);
    }
  }
}

}  // namespace vectorcache
