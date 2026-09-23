#include "vectorcache/cluster/score_centroids.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vectorcache/constants.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

float dot_scalar(const float* a, const float* b, std::size_t dim) {
  float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
  std::size_t d = 0;
  for (; d + 4 <= dim; d += 4) {
    s0 += a[d] * b[d];
    s1 += a[d + 1] * b[d + 1];
    s2 += a[d + 2] * b[d + 2];
    s3 += a[d + 3] * b[d + 3];
  }
  float s = s0 + s1 + s2 + s3;
  for (; d < dim; ++d) s += a[d] * b[d];
  return s;
}

#if defined(__x86_64__) || defined(_M_X64)
bool cpu_has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
  int info[4];
  __cpuid(info, 0);
  if (info[0] >= 7) {
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
  }
  return false;
#else
  return false;
#endif
}

float dot_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t d = 0;
  for (; d + 16 <= dim; d += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + d + 8), _mm256_loadu_ps(b + d + 8), acc1);
  }
  __m256 acc = _mm256_add_ps(acc0, acc1);
  for (; d + 8 <= dim; d += 8) {
    acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + d), _mm256_loadu_ps(b + d), acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; d < dim; ++d) s += a[d] * b[d];
  return s;
}
#endif

float dot_dispatch(const float* a, const float* b, std::size_t dim) {
#if defined(__x86_64__) || defined(_M_X64)
  static const bool use_avx2 = cpu_has_avx2();
  if (use_avx2) return dot_avx2(a, b, dim);
#endif
  return dot_scalar(a, b, dim);
}

float norm_sq_chains(const float* row, std::size_t dim) {
  float chains[kNormChains] = {};
  std::size_t d = 0;
  for (; d + kNormChains <= dim; d += kNormChains) {
    for (std::size_t c = 0; c < kNormChains; ++c) {
      const float v = row[d + c];
      chains[c] += v * v;
    }
  }
  float s = 0.f;
  for (std::size_t c = 0; c < kNormChains; ++c) s += chains[c];
  for (; d < dim; ++d) {
    const float v = row[d];
    s += v * v;
  }
  return s;
}

}  // namespace

void normalize_rows_inplace(std::span<float> rows, std::size_t n_rows, std::size_t dim) {
  if (rows.size() != n_rows * dim) {
    throw std::invalid_argument("normalize_rows_inplace: size mismatch");
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < static_cast<int>(n_rows); ++i) {
    float* row = rows.data() + static_cast<std::size_t>(i) * dim;
    const float nsq = norm_sq_chains(row, dim);
    const float inv = (nsq > kMinInputNorm * kMinInputNorm) ? (1.f / std::sqrt(nsq)) : 0.f;
    for (std::size_t d = 0; d < dim; ++d) row[d] *= inv;
  }
}

void score_against_centroids(std::span<const float> rows, std::size_t n_rows, std::size_t dim,
                             std::span<const float> centroids, std::size_t n_centroids,
                             std::span<float> scores_out) {
  if (rows.size() != n_rows * dim || centroids.size() != n_centroids * dim ||
      scores_out.size() != n_rows * n_centroids) {
    throw std::invalid_argument("score_against_centroids: size mismatch");
  }
  if (n_rows == 0 || n_centroids == 0) return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < static_cast<int>(n_rows); ++i) {
    const float* row = rows.data() + static_cast<std::size_t>(i) * dim;
    float* out = scores_out.data() + static_cast<std::size_t>(i) * n_centroids;
    for (std::size_t j = 0; j < n_centroids; ++j) {
      out[j] = dot_dispatch(row, centroids.data() + j * dim, dim);
    }
  }
}

void assign_nearest_centroid(std::span<const float> rows, std::size_t n_rows, std::size_t dim,
                             std::span<const float> centroids, std::size_t n_centroids,
                             std::span<std::uint32_t> assign_out) {
  if (rows.size() != n_rows * dim || centroids.size() != n_centroids * dim ||
      assign_out.size() != n_rows) {
    throw std::invalid_argument("assign_nearest_centroid: size mismatch");
  }
  if (n_centroids == 0) {
    throw std::invalid_argument("assign_nearest_centroid: need at least one centroid");
  }
  if (n_rows == 0) return;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < static_cast<int>(n_rows); ++i) {
    const float* row = rows.data() + static_cast<std::size_t>(i) * dim;
    float best = -std::numeric_limits<float>::infinity();
    std::uint32_t best_j = 0;
    for (std::size_t j = 0; j < n_centroids; ++j) {
      const float s = dot_dispatch(row, centroids.data() + j * dim, dim);
      if (s > best) {
        best = s;
        best_j = static_cast<std::uint32_t>(j);
      }
    }
    assign_out[static_cast<std::size_t>(i)] = best_j;
  }
}

}  // namespace vectorcache
