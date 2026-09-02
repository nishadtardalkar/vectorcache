#include "vectorcache/transform/fwht.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::transform {

namespace {

constexpr std::size_t kFuseMinStride = 16;

bool is_aligned64(const void* ptr) {
  return (reinterpret_cast<std::uintptr_t>(ptr) & 0x3Fu) == 0;
}

void butterfly_pairs(__m512 a, __m512 b, __m512 scale_v, bool apply_scale, __m512& out_a,
                     __m512& out_b) {
  if (apply_scale) {
    out_a = _mm512_mul_ps(_mm512_add_ps(a, b), scale_v);
    out_b = _mm512_mul_ps(_mm512_sub_ps(a, b), scale_v);
  } else {
    out_a = _mm512_add_ps(a, b);
    out_b = _mm512_sub_ps(a, b);
  }
}

void fwht_stage(float* buf, std::size_t n, std::size_t h, float scale) {
  const bool apply_scale = scale != 1.0f;
  const __m512 scale_v = _mm512_set1_ps(scale);
  const bool aligned = is_aligned64(buf);

  for (std::size_t i = 0; i < n; i += 2 * h) {
    std::size_t j = 0;
    if (aligned && h >= simd::kWidth) {
      for (; j + simd::kWidth <= h; j += simd::kWidth) {
        const __m512 a = _mm512_load_ps(buf + i + j);
        const __m512 b = _mm512_load_ps(buf + i + j + h);
        __m512 out_a;
        __m512 out_b;
        butterfly_pairs(a, b, scale_v, apply_scale, out_a, out_b);
        _mm512_store_ps(buf + i + j, out_a);
        _mm512_store_ps(buf + i + j + h, out_b);
      }
    } else {
      for (; j + simd::kWidth <= h; j += simd::kWidth) {
        const __m512 a = _mm512_loadu_ps(buf + i + j);
        const __m512 b = _mm512_loadu_ps(buf + i + j + h);
        __m512 out_a;
        __m512 out_b;
        butterfly_pairs(a, b, scale_v, apply_scale, out_a, out_b);
        _mm512_storeu_ps(buf + i + j, out_a);
        _mm512_storeu_ps(buf + i + j + h, out_b);
      }
    }

    if (j < h) {
      const __mmask16 mask = simd::tail_mask(h - j);
      const __m512 a = _mm512_maskz_loadu_ps(mask, buf + i + j);
      const __m512 b = _mm512_maskz_loadu_ps(mask, buf + i + j + h);
      __m512 out_a;
      __m512 out_b;
      butterfly_pairs(a, b, scale_v, apply_scale, out_a, out_b);
      _mm512_mask_storeu_ps(buf + i + j, mask, out_a);
      _mm512_mask_storeu_ps(buf + i + j + h, mask, out_b);
    }
  }
}

void fwht_fused_stages(float* buf, std::size_t n, std::size_t h, float scale_2h) {
  const bool apply_scale = scale_2h != 1.0f;
  const __m512 scale_v = _mm512_set1_ps(scale_2h);
  const bool aligned = is_aligned64(buf);

  for (std::size_t i = 0; i < n; i += 4 * h) {
    std::size_t j = 0;
    if (aligned && h >= simd::kWidth) {
      for (; j + simd::kWidth <= h; j += simd::kWidth) {
        __m512 a0 = _mm512_load_ps(buf + i + j);
        __m512 b0 = _mm512_load_ps(buf + i + j + h);
        __m512 c0;
        __m512 d0;
        butterfly_pairs(a0, b0, scale_v, false, c0, d0);
        _mm512_store_ps(buf + i + j, c0);
        _mm512_store_ps(buf + i + j + h, d0);

        __m512 a1 = _mm512_load_ps(buf + i + j + 2 * h);
        __m512 b1 = _mm512_load_ps(buf + i + j + 3 * h);
        __m512 c1;
        __m512 d1;
        butterfly_pairs(a1, b1, scale_v, false, c1, d1);
        _mm512_store_ps(buf + i + j + 2 * h, c1);
        _mm512_store_ps(buf + i + j + 3 * h, d1);

        a0 = _mm512_load_ps(buf + i + j);
        b0 = _mm512_load_ps(buf + i + j + 2 * h);
        butterfly_pairs(a0, b0, scale_v, apply_scale, c0, d0);
        _mm512_store_ps(buf + i + j, c0);
        _mm512_store_ps(buf + i + j + 2 * h, d0);

        a1 = _mm512_load_ps(buf + i + j + h);
        b1 = _mm512_load_ps(buf + i + j + 3 * h);
        butterfly_pairs(a1, b1, scale_v, apply_scale, c1, d1);
        _mm512_store_ps(buf + i + j + h, c1);
        _mm512_store_ps(buf + i + j + 3 * h, d1);
      }
    } else {
      for (; j + simd::kWidth <= h; j += simd::kWidth) {
        __m512 a0 = _mm512_loadu_ps(buf + i + j);
        __m512 b0 = _mm512_loadu_ps(buf + i + j + h);
        __m512 c0;
        __m512 d0;
        butterfly_pairs(a0, b0, scale_v, false, c0, d0);
        _mm512_storeu_ps(buf + i + j, c0);
        _mm512_storeu_ps(buf + i + j + h, d0);

        __m512 a1 = _mm512_loadu_ps(buf + i + j + 2 * h);
        __m512 b1 = _mm512_loadu_ps(buf + i + j + 3 * h);
        __m512 c1;
        __m512 d1;
        butterfly_pairs(a1, b1, scale_v, false, c1, d1);
        _mm512_storeu_ps(buf + i + j + 2 * h, c1);
        _mm512_storeu_ps(buf + i + j + 3 * h, d1);

        a0 = _mm512_loadu_ps(buf + i + j);
        b0 = _mm512_loadu_ps(buf + i + j + 2 * h);
        butterfly_pairs(a0, b0, scale_v, apply_scale, c0, d0);
        _mm512_storeu_ps(buf + i + j, c0);
        _mm512_storeu_ps(buf + i + j + 2 * h, d0);

        a1 = _mm512_loadu_ps(buf + i + j + h);
        b1 = _mm512_loadu_ps(buf + i + j + 3 * h);
        butterfly_pairs(a1, b1, scale_v, apply_scale, c1, d1);
        _mm512_storeu_ps(buf + i + j + h, c1);
        _mm512_storeu_ps(buf + i + j + 3 * h, d1);
      }
    }

    if (j < h) {
      for (std::size_t k = 0; k < h - j; ++k) {
        const std::size_t idx = i + j + k;
        const float a0 = buf[idx];
        const float b0 = buf[idx + h];
        buf[idx] = a0 + b0;
        buf[idx + h] = a0 - b0;

        const float a1 = buf[idx + 2 * h];
        const float b1 = buf[idx + 3 * h];
        buf[idx + 2 * h] = a1 + b1;
        buf[idx + 3 * h] = a1 - b1;
      }
      for (std::size_t k = 0; k < h - j; ++k) {
        const std::size_t idx = i + j + k;
        const float a0 = buf[idx];
        const float b0 = buf[idx + 2 * h];
        if (apply_scale) {
          buf[idx] = (a0 + b0) * scale_2h;
          buf[idx + 2 * h] = (a0 - b0) * scale_2h;
        } else {
          buf[idx] = a0 + b0;
          buf[idx + 2 * h] = a0 - b0;
        }

        const float a1 = buf[idx + h];
        const float b1 = buf[idx + 3 * h];
        if (apply_scale) {
          buf[idx + h] = (a1 + b1) * scale_2h;
          buf[idx + 3 * h] = (a1 - b1) * scale_2h;
        } else {
          buf[idx + h] = a1 + b1;
          buf[idx + 3 * h] = a1 - b1;
        }
      }
    }
  }
}

void fwht_orthonormal_impl(float* buf, std::size_t n, float inv_sqrt_n) {
  std::size_t h = 1;
  while (h < n) {
    if (h >= kFuseMinStride && h * 4 <= n) {
      const bool is_last = (h * 4 == n);
      const float scale = is_last ? inv_sqrt_n : 1.0f;
      fwht_fused_stages(buf, n, h, scale);
      h *= 4;
    } else {
      const bool is_last = (h * 2 == n);
      const float scale = is_last ? inv_sqrt_n : 1.0f;
      fwht_stage(buf, n, h, scale);
      h *= 2;
    }
  }
}

void fwht_orthonormal_unfused_impl(float* buf, std::size_t n, float inv_sqrt_n) {
  std::size_t h = 1;
  while (h < n) {
    const bool is_last = (h * 2 == n);
    const float scale = is_last ? inv_sqrt_n : 1.0f;
    fwht_stage(buf, n, h, scale);
    h *= 2;
  }
}

void fwht_256_orthonormal(float* buf, float inv_sqrt_n) {
  fwht_stage(buf, 256, 1, 1.0f);
  fwht_stage(buf, 256, 2, 1.0f);
  fwht_stage(buf, 256, 4, 1.0f);
  fwht_stage(buf, 256, 8, 1.0f);
  fwht_fused_stages(buf, 256, 16, 1.0f);
  fwht_fused_stages(buf, 256, 64, inv_sqrt_n);
}

void fwht_2048_orthonormal(float* buf, float inv_sqrt_n) {
  fwht_orthonormal_impl(buf, 2048, inv_sqrt_n);
}

void fwht_4096_orthonormal(float* buf, float inv_sqrt_n) {
  fwht_orthonormal_impl(buf, 4096, inv_sqrt_n);
}

void dispatch_orthonormal(float* buf, std::size_t n, float inv_sqrt_n) {
  switch (n) {
    case 256:
      fwht_256_orthonormal(buf, inv_sqrt_n);
      return;
    case 2048:
      fwht_2048_orthonormal(buf, inv_sqrt_n);
      return;
    case 4096:
      fwht_4096_orthonormal(buf, inv_sqrt_n);
      return;
    default:
      fwht_orthonormal_impl(buf, n, inv_sqrt_n);
      return;
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

  fwht_orthonormal_impl(buf.data(), n, 1.0f);
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
  const std::size_t n = buf.size();
  if (n == 0) {
    throw Error("FWHT buffer must be non-empty");
  }
  if ((n & (n - 1)) != 0) {
    throw Error("FWHT buffer length must be a power of two, got " + std::to_string(n));
  }
  dispatch_orthonormal(buf.data(), n, inv_sqrt_n);
}

void fwht_stockham_orthonormal_in_place(std::span<float> buf, std::span<float> scratch,
                                        float inv_sqrt_n) {
  (void)scratch;
  fwht_orthonormal_in_place(buf, inv_sqrt_n);
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

void fwht_orthonormal_unfused_in_place(std::span<float> buf, float inv_sqrt_n) {
  const std::size_t n = buf.size();
  if (n == 0) {
    throw Error("FWHT buffer must be non-empty");
  }
  if ((n & (n - 1)) != 0) {
    throw Error("FWHT buffer length must be a power of two, got " + std::to_string(n));
  }
  fwht_orthonormal_unfused_impl(buf.data(), n, inv_sqrt_n);
}

void apply_signs_f(std::span<float> buf, std::span<const float> signs) {
  const std::size_t len = buf.size();
  const bool aligned =
      is_aligned64(buf.data()) && is_aligned64(signs.data());
  std::size_t i = 0;
  if (aligned) {
    for (; i + simd::kWidth <= len; i += simd::kWidth) {
      const __m512 v = _mm512_load_ps(buf.data() + i);
      const __m512 s = _mm512_load_ps(signs.data() + i);
      _mm512_store_ps(buf.data() + i, _mm512_mul_ps(v, s));
    }
  } else {
    for (; i + simd::kWidth <= len; i += simd::kWidth) {
      const __m512 v = _mm512_loadu_ps(buf.data() + i);
      const __m512 s = _mm512_loadu_ps(signs.data() + i);
      _mm512_storeu_ps(buf.data() + i, _mm512_mul_ps(v, s));
    }
  }
  if (i < len) {
    const __mmask16 mask = simd::tail_mask(len - i);
    const __m512 v = _mm512_maskz_loadu_ps(mask, buf.data() + i);
    const __m512 s = _mm512_maskz_loadu_ps(mask, signs.data() + i);
    _mm512_mask_storeu_ps(buf.data() + i, mask, _mm512_mul_ps(v, s));
  }
}

}  // namespace vectorcache::transform
