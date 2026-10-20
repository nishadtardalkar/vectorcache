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

/// Signed 4-bit levels in [-7, 7].
inline std::int8_t quantize_to_s4(float x, float inv_scale) {
  const float q = std::round(x * inv_scale);
  const float clamped = std::max(-7.f, std::min(7.f, q));
  return static_cast<std::int8_t>(clamped);
}

inline std::int8_t quantize_to_i8(float x, float inv_scale) {
  const float q = std::round(x * inv_scale);
  const float clamped = std::max(-127.f, std::min(127.f, q));
  return static_cast<std::int8_t>(clamped);
}

inline std::uint8_t pack_s4_pair(std::int8_t even, std::int8_t odd) {
  return static_cast<std::uint8_t>((static_cast<std::uint8_t>(odd + 7) << 4) |
                                   static_cast<std::uint8_t>(even + 7));
}

inline void unpack_s4_pair(std::uint8_t b, std::int8_t& even, std::int8_t& odd) {
  even = static_cast<std::int8_t>((b & 0x0Fu) - 7);
  odd = static_cast<std::int8_t>((b >> 4) - 7);
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

/// Dot q_i8[dim_pad] with 4-bit packed L column (dim_pad/2 bytes). Pad must be 0.
inline std::int32_t dot_i8_q4_avx2(const std::int8_t* q, const std::uint8_t* packed,
                                   std::size_t dim_pad) {
  const __m128i k7 = _mm_set1_epi8(7);
  const __m128i k0f = _mm_set1_epi8(0x0f);
  __m256i acc = _mm256_setzero_si256();
  for (std::size_t i = 0, b = 0; i < dim_pad; i += 16, b += 8) {
    // 8 packed bytes → 16 signed nibbles (even=lo, odd=hi).
    const __m128i pk = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed + b));
    const __m128i even_u = _mm_and_si128(pk, k0f);
    const __m128i odd_u = _mm_and_si128(_mm_srli_epi16(pk, 4), k0f);
    const __m128i even_s = _mm_sub_epi8(even_u, k7);
    const __m128i odd_s = _mm_sub_epi8(odd_u, k7);
    // Interleave even/odd bytes into dim order: e0,o0,e1,o1,...
    const __m128i lo = _mm_unpacklo_epi8(even_s, odd_s);  // 16 int8 values
    const __m256i Lw = _mm256_cvtepi8_epi16(lo);
    const __m128i q16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q + i));
    const __m256i qw = _mm256_cvtepi8_epi16(q16);
    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(qw, Lw));
  }
  return hsum256_epi32(acc);
}
#endif

inline std::int32_t dot_i8_q4_scalar(const std::int8_t* q, const std::uint8_t* packed,
                                     std::size_t dim) {
  std::int32_t acc = 0;
  const std::size_t n_bytes = (dim + 1u) / 2u;
  for (std::size_t b = 0; b < n_bytes; ++b) {
    std::int8_t e = 0;
    std::int8_t o = 0;
    unpack_s4_pair(packed[b], e, o);
    const std::size_t i0 = 2 * b;
    acc += static_cast<std::int32_t>(q[i0]) * static_cast<std::int32_t>(e);
    if (i0 + 1 < dim) {
      acc += static_cast<std::int32_t>(q[i0 + 1]) * static_cast<std::int32_t>(o);
    }
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
  const std::size_t bytes_per_col = out.dim_pad / 2;
  out.codes.assign(bytes_per_col * m, 0);

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
    out.scales[j] = amax / 7.f;
    const float inv = 7.f / amax;
    std::uint8_t* dest = out.codes.data() + j * bytes_per_col;
    for (std::size_t i = 0; i < dim; i += 2) {
      const std::int8_t e = quantize_to_s4(col[i], inv);
      const std::int8_t o = (i + 1 < dim) ? quantize_to_s4(col[i + 1], inv) : 0;
      dest[i / 2] = pack_s4_pair(e, o);
    }
  }
  return out;
}

void project_onto_landmarks_q4(const float* q_rot, std::size_t nq, std::size_t dim,
                               const QuantizedLandmarks& L, float* q_land, std::size_t scan_dim) {
  if (L.dim != dim || L.m == 0 || scan_dim < L.m) {
    throw std::invalid_argument("project_onto_landmarks_q4: shape mismatch");
  }
  const std::size_t m = L.m;
  const std::size_t dim_pad = L.dim_pad;
  const std::size_t bytes_per_col = dim_pad / 2;
  const std::uint8_t* codes = L.codes.data();
  const float* scales = L.scales.data();

  // Query stays int8 (asymmetric); L is 4-bit packed.
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
        const std::uint8_t* Lj = codes + (j0 + t) * bytes_per_col;
#if defined(__x86_64__) || defined(_M_X64)
        const std::int32_t dot = dot_i8_q4_avx2(qrow, Lj, dim_pad);
#else
        const std::int32_t dot = dot_i8_q4_scalar(qrow, Lj, dim);
#endif
        dest[t] = qs * scales[j0 + t] * static_cast<float>(dot);
      }
    }
  }

  for (std::size_t j = m_aligned; j < m; ++j) {
    const std::uint8_t* Lj = codes + j * bytes_per_col;
    const float ls = scales[j];
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::int8_t* qrow = q_i8.data() + qis * dim_pad;
#if defined(__x86_64__) || defined(_M_X64)
      const std::int32_t dot = dot_i8_q4_avx2(qrow, Lj, dim_pad);
#else
      const std::int32_t dot = dot_i8_q4_scalar(qrow, Lj, dim);
#endif
      q_land[qis * scan_dim + j] = q_scales[qis] * ls * static_cast<float>(dot);
    }
  }
}

}  // namespace vectorcache
