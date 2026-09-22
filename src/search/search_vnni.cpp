#include "vectorcache/search/search_vnni.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

#include "vectorcache/pack/pack.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vectorcache {
namespace {

void heap_push_or_replace(float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                          float& heap_min, std::size_t& heap_mi, std::size_t k, float score,
                          std::uint64_t id) {
  if (heap_sz < k) {
    heap_s[heap_sz] = score;
    heap_i[heap_sz] = id;
    ++heap_sz;
    if (heap_sz == k) {
      heap_min = heap_s[0];
      heap_mi = 0;
      for (std::size_t i = 1; i < k; ++i) {
        if (heap_s[i] < heap_min) {
          heap_min = heap_s[i];
          heap_mi = i;
        }
      }
    }
  } else if (score > heap_min) {
    heap_s[heap_mi] = score;
    heap_i[heap_mi] = id;
    heap_min = heap_s[0];
    heap_mi = 0;
    for (std::size_t i = 1; i < k; ++i) {
      if (heap_s[i] < heap_min) {
        heap_min = heap_s[i];
        heap_mi = i;
      }
    }
  }
}

#if defined(__x86_64__) || defined(_M_X64)

void flush_block_heap(__m512 f0, __m512 f1, std::size_t base_vec, std::size_t n_vectors,
                      const float* vec_scales, std::size_t k, float* heap_s, std::uint64_t* heap_i,
                      std::size_t& heap_sz, float& heap_min, std::size_t& heap_mi) {
  alignas(64) float scores[kBlock];
  _mm512_storeu_ps(scores, f0);
  _mm512_storeu_ps(scores + 16, f1);
  const std::size_t end = std::min(base_vec + kBlock, n_vectors);
  for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
    const std::size_t vi = base_vec + lane;
    heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k,
                         scores[lane] * vec_scales[vi], static_cast<std::uint64_t>(vi));
  }
}

void score_query_vnni_impl(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                           std::span<const float> vec_scales, std::size_t n_byte_groups,
                           std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                           std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                           std::size_t& heap_mi, float bias_corr) {
  assert(n_byte_groups % 4 == 0);
  assert(lut.uint8_luts != nullptr);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const __m512i kpos = _mm512_set1_epi32(static_cast<int>(0x30201000u));
  const __m512i ones = _mm512_set1_epi8(1);
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const float bias = lut.bias + bias_corr;
  const __m512 vs = _mm512_set1_ps(lut.scale);
  const __m512 vb = _mm512_set1_ps(bias);

  const std::size_t n_pairs = n_blocks / 2;
  for (std::size_t pb = 0; pb < n_pairs; ++pb) {
    const std::size_t b = pb * 2;
    const std::size_t base0 = b * block_bytes;
    const std::size_t base1 = base0 + block_bytes;
    __m512i a0[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
    __m512i a1[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};

    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const std::size_t pf = base0 + (q4 + 8) * 128 + h * 64;
        if (pf + 64 <= blocked_codes.size()) {
#if defined(_MSC_VER)
          _mm_prefetch(reinterpret_cast<const char*>(blocked_codes.data() + pf), _MM_HINT_T0);
#else
          __builtin_prefetch(blocked_codes.data() + pf, 0, 3);
#endif
        }
        const std::uint8_t* tp = lut.uint8_luts + q4 * 128;
        const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp));
        const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp + 64));
        const __m512i c0 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base0 + q4 * 128 + h * 64));
        const __m512i c1 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base1 + q4 * 128 + h * 64));
        const __m512i i0lo = _mm512_or_si512(_mm512_and_si512(c0, m0f), kpos);
        const __m512i i0hi =
            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c0, 4), m0f), kpos);
        const __m512i i1lo = _mm512_or_si512(_mm512_and_si512(c1, m0f), kpos);
        const __m512i i1hi =
            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c1, 4), m0f), kpos);
        a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0lo, tlo), ones);
        a0[h] = _mm512_dpbusd_epi32(a0[h], _mm512_permutexvar_epi8(i0hi, thi), ones);
        a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1lo, tlo), ones);
        a1[h] = _mm512_dpbusd_epi32(a1[h], _mm512_permutexvar_epi8(i1hi, thi), ones);
      }
    }

    for (std::size_t i = 0; i < 2; ++i) {
      const __m512i* a = (i == 0) ? a0 : a1;
      const std::size_t base_vec = (b + i) * kBlock;
      const __m512 f0 =
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
      const __m512 f1 =
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                       heap_min, heap_mi);
    }
  }

  for (std::size_t b = n_pairs * 2; b < n_blocks; ++b) {
    const std::size_t base = b * block_bytes;
    __m512i a[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const std::uint8_t* tp = lut.uint8_luts + q4 * 128;
        const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp));
        const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp + 64));
        const __m512i c = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base + q4 * 128 + h * 64));
        const __m512i ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
        const __m512i ihi =
            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
        a[h] = _mm512_dpbusd_epi32(a[h], _mm512_permutexvar_epi8(ilo, tlo), ones);
        a[h] = _mm512_dpbusd_epi32(a[h], _mm512_permutexvar_epi8(ihi, thi), ones);
      }
    }
    const std::size_t base_vec = b * kBlock;
    const __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
    const __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
    flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                     heap_min, heap_mi);
  }
}

#endif  // x86_64

}  // namespace

std::vector<std::uint8_t> split_lut_for_vnni(std::span<const std::uint8_t> uint8_luts,
                                            std::size_t n_byte_groups) {
  assert(uint8_luts.size() == n_byte_groups * 32);
  assert(n_byte_groups % 4 == 0);
  std::vector<std::uint8_t> out(n_byte_groups * 32);
  for (std::size_t g0 = 0; g0 < n_byte_groups; g0 += 4) {
    const std::size_t c = (g0 / 4) * 128;
    for (std::size_t j = 0; j < 4; ++j) {
      const std::size_t src = (g0 + j) * 32;
      // uint8_luts is [hi_16 | lo_16] per group; VNNI wants lo then hi.
      std::memcpy(out.data() + c + j * 16, uint8_luts.data() + src + 16, 16);
      std::memcpy(out.data() + c + 64 + j * 16, uint8_luts.data() + src, 16);
    }
  }
  return out;
}

void score_query_vnni(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                      std::span<const float> vec_scales, std::size_t n_byte_groups,
                      std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                      std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                      std::size_t& heap_mi, float bias_corr) {
#if defined(__x86_64__) || defined(_M_X64)
  score_query_vnni_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                        heap_s, heap_i, heap_sz, heap_min, heap_mi, bias_corr);
#else
  (void)lut;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)k;
  (void)heap_s;
  (void)heap_i;
  (void)heap_sz;
  (void)heap_min;
  (void)heap_mi;
  (void)bias_corr;
  throw std::logic_error("score_query_vnni is only available on x86_64");
#endif
}

}  // namespace vectorcache
