#include "vectorcache/transform/fwht.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::transform {

namespace {

void butterfly_block(std::span<float> buf, std::size_t i, std::size_t h, float scale = 1.0f) {
  const bool apply_scale = scale != 1.0f;
  const __m512 scale_v = _mm512_set1_ps(scale);

  if (h >= simd::kWidth) {
    for (std::size_t j = i; j < i + h; ) {
      const std::size_t chunk = std::min(simd::kWidth, i + h - j);
      if (chunk == simd::kWidth) {
        const __m512 a = _mm512_loadu_ps(buf.data() + j);
        const __m512 b = _mm512_loadu_ps(buf.data() + j + h);
        if (apply_scale) {
          _mm512_storeu_ps(buf.data() + j, _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
          _mm512_storeu_ps(buf.data() + j + h, _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
        } else {
          _mm512_storeu_ps(buf.data() + j, _mm512_add_ps(a, b));
          _mm512_storeu_ps(buf.data() + j + h, _mm512_sub_ps(a, b));
        }
      } else {
        const __mmask16 mask = simd::tail_mask(chunk);
        const __m512 a = _mm512_maskz_loadu_ps(mask, buf.data() + j);
        const __m512 b = _mm512_maskz_loadu_ps(mask, buf.data() + j + h);
        if (apply_scale) {
          _mm512_mask_storeu_ps(buf.data() + j, mask,
                                _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
          _mm512_mask_storeu_ps(buf.data() + j + h, mask,
                                _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
        } else {
          _mm512_mask_storeu_ps(buf.data() + j, mask, _mm512_add_ps(a, b));
          _mm512_mask_storeu_ps(buf.data() + j + h, mask, _mm512_sub_ps(a, b));
        }
      }
      j += chunk;
    }
    return;
  }

  const __mmask16 mask = simd::tail_mask(h);
  const __m512 a = _mm512_maskz_loadu_ps(mask, buf.data() + i);
  const __m512 b = _mm512_maskz_loadu_ps(mask, buf.data() + i + h);
  if (apply_scale) {
    _mm512_mask_storeu_ps(buf.data() + i, mask, _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
    _mm512_mask_storeu_ps(buf.data() + i + h, mask, _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
  } else {
    _mm512_mask_storeu_ps(buf.data() + i, mask, _mm512_add_ps(a, b));
    _mm512_mask_storeu_ps(buf.data() + i + h, mask, _mm512_sub_ps(a, b));
  }
}

void butterfly_block_ping_pong(const float* src, float* dst, std::size_t i, std::size_t h,
                               float scale) {
  const bool apply_scale = scale != 1.0f;
  const __m512 scale_v = _mm512_set1_ps(scale);

  if (h >= simd::kWidth) {
    for (std::size_t j = i; j < i + h; ) {
      const std::size_t chunk = std::min(simd::kWidth, i + h - j);
      if (chunk == simd::kWidth) {
        const __m512 a = _mm512_loadu_ps(src + j);
        const __m512 b = _mm512_loadu_ps(src + j + h);
        if (apply_scale) {
          _mm512_storeu_ps(dst + j, _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
          _mm512_storeu_ps(dst + j + h, _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
        } else {
          _mm512_storeu_ps(dst + j, _mm512_add_ps(a, b));
          _mm512_storeu_ps(dst + j + h, _mm512_sub_ps(a, b));
        }
      } else {
        const __mmask16 mask = simd::tail_mask(chunk);
        const __m512 a = _mm512_maskz_loadu_ps(mask, src + j);
        const __m512 b = _mm512_maskz_loadu_ps(mask, src + j + h);
        if (apply_scale) {
          _mm512_mask_storeu_ps(dst + j, mask, _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
          _mm512_mask_storeu_ps(dst + j + h, mask, _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
        } else {
          _mm512_mask_storeu_ps(dst + j, mask, _mm512_add_ps(a, b));
          _mm512_mask_storeu_ps(dst + j + h, mask, _mm512_sub_ps(a, b));
        }
      }
      j += chunk;
    }
    return;
  }

  const __mmask16 mask = simd::tail_mask(h);
  const __m512 a = _mm512_maskz_loadu_ps(mask, src + i);
  const __m512 b = _mm512_maskz_loadu_ps(mask, src + i + h);
  if (apply_scale) {
    _mm512_mask_storeu_ps(dst + i, mask, _mm512_mul_ps(_mm512_add_ps(a, b), scale_v));
    _mm512_mask_storeu_ps(dst + i + h, mask, _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v));
  } else {
    _mm512_mask_storeu_ps(dst + i, mask, _mm512_add_ps(a, b));
    _mm512_mask_storeu_ps(dst + i + h, mask, _mm512_sub_ps(a, b));
  }
}

void fwht_in_place_impl(std::span<float> buf, float last_stage_scale) {
  const std::size_t n = buf.size();
  std::size_t h = 1;
  while (h < n) {
    const bool is_last = (h * 2 == n);
    const float stage_scale = is_last ? last_stage_scale : 1.0f;
    for (std::size_t i = 0; i < n; i += 2 * h) {
      butterfly_block(buf, i, h, stage_scale);
    }
    h *= 2;
  }
}

}  // namespace

std::size_t padded_dim(std::size_t dim) {
  if (dim == 0) {
    return 1;
  }
  std::size_t n = 1;
  while (n < dim) {
    n <<= 1;
  }
  return n;
}

void fwht_in_place(std::span<float> buf) {
  const std::size_t n = buf.size();
  if (n == 0) {
    throw Error("FWHT buffer must be non-empty");
  }
  if ((n & (n - 1)) != 0) {
    throw Error("FWHT buffer length must be a power of two, got " + std::to_string(n));
  }

  fwht_in_place_impl(buf, 1.0f);
}

void scale_in_place(std::span<float> buf, float scale) {
  const __m512 scale_v = _mm512_set1_ps(scale);
  std::size_t i = 0;
  const std::size_t n = buf.size();
  for (; i + simd::kWidth <= n; i += simd::kWidth) {
    __m512 v = _mm512_loadu_ps(buf.data() + i);
    v = _mm512_mul_ps(v, scale_v);
    _mm512_storeu_ps(buf.data() + i, v);
  }
  if (i < n) {
    const __mmask16 mask = simd::tail_mask(n - i);
    const __m512 v = _mm512_maskz_loadu_ps(mask, buf.data() + i);
    _mm512_mask_storeu_ps(buf.data() + i, mask, _mm512_mul_ps(v, scale_v));
  }
}

void fwht_orthonormal_in_place(std::span<float> buf) {
  fwht_orthonormal_in_place(buf, 1.0f / std::sqrt(static_cast<float>(buf.size())));
}

void fwht_orthonormal_in_place(std::span<float> buf, float inv_sqrt_n) {
  fwht_in_place_impl(buf, inv_sqrt_n);
}

void fwht_stockham_orthonormal_in_place(std::span<float> buf, std::span<float> scratch,
                                        float inv_sqrt_n) {
  const std::size_t n = buf.size();
  if (scratch.size() != n) {
    throw Error("FWHT scratch buffer size mismatch");
  }
  if (n == 0) {
    throw Error("FWHT buffer must be non-empty");
  }
  if ((n & (n - 1)) != 0) {
    throw Error("FWHT buffer length must be a power of two, got " + std::to_string(n));
  }

  float* src = buf.data();
  float* dst = scratch.data();

  for (std::size_t h = 1; h < n; h *= 2) {
    const bool is_last = (h * 2 == n);
    const float stage_scale = is_last ? inv_sqrt_n : 1.0f;
    for (std::size_t i = 0; i < n; i += 2 * h) {
      butterfly_block_ping_pong(src, dst, i, h, stage_scale);
    }
    float* const tmp = src;
    src = dst;
    dst = tmp;
  }

  if (src != buf.data()) {
    std::memcpy(buf.data(), src, n * sizeof(float));
  }
}

void apply_signs_i8(std::span<float> buf, std::span<const std::int8_t> signs) {
  const std::size_t len = buf.size();
  const __m512 zero = _mm512_setzero_ps();
  std::size_t i = 0;
  while (i + simd::kWidth <= len) {
    const __m128i sign_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(signs.data() + i));
    const __m512i sign_i32 = _mm512_cvtepi8_epi32(sign_bytes);
    const __mmask16 neg = _mm512_cmplt_epi32_mask(sign_i32, _mm512_setzero_si512());
    const __m512 v = _mm512_loadu_ps(buf.data() + i);
    _mm512_storeu_ps(buf.data() + i, _mm512_mask_sub_ps(v, neg, zero, v));
    i += simd::kWidth;
  }
  if (i < len) {
    const __mmask16 mask = simd::tail_mask(len - i);
    alignas(16) std::int8_t sign_buf[simd::kWidth] = {};
    std::memcpy(sign_buf, signs.data() + i, len - i);
    const __m128i sign_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(sign_buf));
    const __m512i sign_i32 = _mm512_cvtepi8_epi32(sign_bytes);
    const __mmask16 neg = _mm512_cmplt_epi32_mask(sign_i32, _mm512_setzero_si512());
    const __m512 v = _mm512_maskz_loadu_ps(mask, buf.data() + i);
    _mm512_mask_storeu_ps(buf.data() + i, mask, _mm512_mask_sub_ps(v, neg, zero, v));
  }
}

}  // namespace vectorcache::transform
