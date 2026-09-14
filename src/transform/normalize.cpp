#include "vectorcache/transform/normalize.hpp"

#include <cmath>
#include <cstdint>

#include "vectorcache/simd.hpp"

namespace vectorcache::transform {

namespace {

bool is_aligned64(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 63u) == 0;
}

float norm_sq_simd(std::span<const float> vector) {
  __m512 sum = _mm512_setzero_ps();
  std::size_t i = 0;
  const std::size_t n = vector.size();
  const bool aligned = is_aligned64(vector.data());
  if (aligned) {
    for (; i + simd::kWidth <= n; i += simd::kWidth) {
      const __m512 v = _mm512_load_ps(vector.data() + i);
      sum = _mm512_fmadd_ps(v, v, sum);
    }
  } else {
    for (; i + simd::kWidth <= n; i += simd::kWidth) {
      const __m512 v = _mm512_loadu_ps(vector.data() + i);
      sum = _mm512_fmadd_ps(v, v, sum);
    }
  }
  float norm_sq = _mm512_reduce_add_ps(sum);
  if (i < n) {
    const __mmask16 mask = simd::tail_mask(n - i);
    const __m512 v = _mm512_maskz_loadu_ps(mask, vector.data() + i);
    const __m512 partial = _mm512_maskz_fmadd_ps(mask, v, v, _mm512_setzero_ps());
    norm_sq += _mm512_reduce_add_ps(partial);
  }
  return norm_sq;
}

}  // namespace

void l2_normalize_in_place(std::span<float> vector) {
  const float norm_sq = norm_sq_simd(vector);
  if (norm_sq > 0.0f) {
    const float inv = 1.0f / std::sqrt(norm_sq);
    const __m512 inv_v = _mm512_set1_ps(inv);
    std::size_t i = 0;
    const std::size_t n = vector.size();
    const bool aligned = is_aligned64(vector.data());
    if (aligned) {
      for (; i + simd::kWidth <= n; i += simd::kWidth) {
        __m512 v = _mm512_load_ps(vector.data() + i);
        v = _mm512_mul_ps(v, inv_v);
        _mm512_store_ps(vector.data() + i, v);
      }
    } else {
      for (; i + simd::kWidth <= n; i += simd::kWidth) {
        __m512 v = _mm512_loadu_ps(vector.data() + i);
        v = _mm512_mul_ps(v, inv_v);
        _mm512_storeu_ps(vector.data() + i, v);
      }
    }
    if (i < n) {
      const __mmask16 mask = simd::tail_mask(n - i);
      const __m512 v = _mm512_maskz_loadu_ps(mask, vector.data() + i);
      _mm512_mask_storeu_ps(vector.data() + i, mask, _mm512_mul_ps(v, inv_v));
    }
  }
}

}  // namespace vectorcache::transform
