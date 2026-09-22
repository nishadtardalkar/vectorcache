#include "vectorcache/search/search_vnni.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vectorcache/pack/pack.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vectorcache {
namespace {

void rescan_min(const float* heap_s, const std::uint64_t* heap_i, std::size_t k, float& heap_min,
                std::size_t& heap_mi) {
  heap_mi = 0;
  heap_min = heap_s[0];
  for (std::size_t i = 1; i < k; ++i) {
    if (heap_s[i] < heap_min || (heap_s[i] == heap_min && heap_i[i] > heap_i[heap_mi])) {
      heap_min = heap_s[i];
      heap_mi = i;
    }
  }
}

void heap_push_or_replace(float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                          float& heap_min, std::size_t& heap_mi, std::size_t k, float score,
                          std::uint64_t id) {
  if (heap_sz < k) {
    heap_s[heap_sz] = score;
    heap_i[heap_sz] = id;
    ++heap_sz;
    if (heap_sz == k) {
      rescan_min(heap_s, heap_i, k, heap_min, heap_mi);
    }
  } else if (score > heap_min) {
    heap_s[heap_mi] = score;
    heap_i[heap_mi] = id;
    rescan_min(heap_s, heap_i, k, heap_min, heap_mi);
  }
}

#if defined(__x86_64__) || defined(_M_X64)

// On non-Windows, keep multi-query kernels out-of-line. MinGW Win64 has a
// ZMM spill-alignment bug with some attributes; leave kernels unadorned there.
#if defined(__GNUC__) || defined(__clang__)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
#define VECTORCACHE_AVX512_FN
#else
#define VECTORCACHE_AVX512_FN __attribute__((noinline, noipa))
#endif
#else
#define VECTORCACHE_AVX512_FN
#endif

VECTORCACHE_AVX512_FN
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

VECTORCACHE_AVX512_FN
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
      const __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[0]), vs), vb);
      const __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a[1]), vs), vb);
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

#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
VECTORCACHE_AVX512_FN
void score_queries_vnni_multi(const std::uint8_t* const* split_luts, const float* lut_scales,
                              const float* lut_biases, std::size_t nq,
                              std::span<const std::uint8_t> blocked_codes,
                              std::span<const float> vec_scales, std::size_t n_byte_groups,
                              std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                              float* const* heap_s, std::uint64_t* const* heap_i,
                              std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi) {
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= 8);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const __m512i kpos = _mm512_set1_epi32(static_cast<int>(0x30201000u));
  const __m512i ones = _mm512_set1_epi8(1);
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[8][2];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      acc[qi][0] = _mm512_setzero_si512();
      acc[qi][1] = _mm512_setzero_si512();
    }

    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const __m512i c = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + block_base + q4 * 128 + h * 64));
        const __m512i ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
        const __m512i ihi =
            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          const std::uint8_t* tp = split_luts[qi] + q4 * 128;
          const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp));
          const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp + 64));
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], _mm512_permutexvar_epi8(ilo, tlo), ones);
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], _mm512_permutexvar_epi8(ihi, thi), ones);
        }
      }
    }

    for (std::size_t qi = 0; qi < nq; ++qi) {
      const __m512 vs = _mm512_set1_ps(lut_scales[qi]);
      const __m512 vb = _mm512_set1_ps(lut_biases[qi]);
      const __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs), vb);
      const __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs), vb);
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s[qi], heap_i[qi],
                       heap_sz[qi], heap_min[qi], heap_mi[qi]);
    }
  }
}
#endif  // !MinGW multi VNNI

#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
VECTORCACHE_AVX512_FN
void score_query_permute_dot_blk2(const QueryPermuteDot& pd,
                                  std::span<const std::uint8_t> blocked_codes,
                                  std::span<const float> vec_scales, std::size_t n_byte_groups,
                                  std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                  float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                  float& heap_min, std::size_t& heap_mi) {
  assert(n_byte_groups % 4 == 0);
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pd.levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));

  const std::size_t n_pairs = n_blocks / 2;
  for (std::size_t pb = 0; pb < n_pairs; ++pb) {
    const std::size_t b = pb * 2;
    const std::size_t base0 = b * block_bytes;
    const std::size_t base1 = base0 + block_bytes;
    const __m512i z = _mm512_set1_epi32(pd.zero);
    __m512i a0[2] = {z, z};
    __m512i a1[2] = {z, z};

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
        const __m512i c0 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base0 + q4 * 128 + h * 64));
        const __m512i c1 = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base1 + q4 * 128 + h * 64));
        const __m512i vlo0 = _mm512_shuffle_epi8(levels, _mm512_and_si512(c0, m0f));
        const __m512i vhi0 =
            _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c0, 4), m0f));
        const __m512i vlo1 = _mm512_shuffle_epi8(levels, _mm512_and_si512(c1, m0f));
        const __m512i vhi1 =
            _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c1, 4), m0f));
        const std::int8_t* wp = pd.weights.data() + q4 * 8;
        std::int32_t wlo_i = 0;
        std::int32_t whi_i = 0;
        std::memcpy(&wlo_i, wp, 4);
        std::memcpy(&whi_i, wp + 4, 4);
        const __m512i wlo = _mm512_set1_epi32(wlo_i);
        const __m512i whi = _mm512_set1_epi32(whi_i);
        a0[h] = _mm512_dpbusd_epi32(a0[h], vlo0, wlo);
        a0[h] = _mm512_dpbusd_epi32(a0[h], vhi0, whi);
        a1[h] = _mm512_dpbusd_epi32(a1[h], vlo1, wlo);
        a1[h] = _mm512_dpbusd_epi32(a1[h], vhi1, whi);
      }
    }

    const __m512 vs = _mm512_set1_ps(pd.scale);
    const __m512 vb = _mm512_set1_ps(pd.bias);
    for (std::size_t i = 0; i < 2; ++i) {
      const __m512i* a = (i == 0) ? a0 : a1;
      const std::size_t base_vec = (b + i) * kBlock;
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1]), vs, vb);
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                       heap_min, heap_mi);
    }
  }

  for (std::size_t b = n_pairs * 2; b < n_blocks; ++b) {
    const std::size_t base = b * block_bytes;
    const __m512i z = _mm512_set1_epi32(pd.zero);
    __m512i a[2] = {z, z};
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const __m512i c = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + base + q4 * 128 + h * 64));
        const __m512i vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
        const __m512i vhi =
            _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f));
        const std::int8_t* wp = pd.weights.data() + q4 * 8;
        std::int32_t wlo_i = 0;
        std::int32_t whi_i = 0;
        std::memcpy(&wlo_i, wp, 4);
        std::memcpy(&whi_i, wp + 4, 4);
        a[h] = _mm512_dpbusd_epi32(a[h], vlo, _mm512_set1_epi32(wlo_i));
        a[h] = _mm512_dpbusd_epi32(a[h], vhi, _mm512_set1_epi32(whi_i));
      }
    }
    const __m512 vs = _mm512_set1_ps(pd.scale);
    const __m512 vb = _mm512_set1_ps(pd.bias);
    const std::size_t base_vec = b * kBlock;
    const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0]), vs, vb);
    const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1]), vs, vb);
    flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                     heap_min, heap_mi);
  }
}

VECTORCACHE_AVX512_FN
void score_queries_permute_dot_multi(const QueryPermuteDot* const* pds, std::size_t nq,
                                     std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                     float* const* heap_s, std::uint64_t* const* heap_i,
                                     std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi) {
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= 8);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(pds[0]->levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[8][2];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      const __m512i z = _mm512_set1_epi32(pds[qi]->zero);
      acc[qi][0] = z;
      acc[qi][1] = z;
    }

    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const std::size_t pf = block_base + (q4 + 32) * 128 + h * 64;
        if (pf + 64 <= blocked_codes.size()) {
#if defined(_MSC_VER)
          _mm_prefetch(reinterpret_cast<const char*>(blocked_codes.data() + pf), _MM_HINT_T0);
#else
          __builtin_prefetch(blocked_codes.data() + pf, 0, 3);
#endif
        }
        const __m512i c = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + block_base + q4 * 128 + h * 64));
        const __m512i vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
        const __m512i vhi =
            _mm512_shuffle_epi8(levels, _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f));
        for (std::size_t qi = 0; qi < nq; ++qi) {
          const std::int8_t* wp = pds[qi]->weights.data() + q4 * 8;
          std::int32_t wlo_i = 0;
          std::int32_t whi_i = 0;
          std::memcpy(&wlo_i, wp, 4);
          std::memcpy(&whi_i, wp + 4, 4);
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vlo, _mm512_set1_epi32(wlo_i));
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vhi, _mm512_set1_epi32(whi_i));
        }
      }
    }

    for (std::size_t qi = 0; qi < nq; ++qi) {
      const __m512 vs = _mm512_set1_ps(pds[qi]->scale);
      const __m512 vb = _mm512_set1_ps(pds[qi]->bias);
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs, vb);
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s[qi], heap_i[qi],
                       heap_sz[qi], heap_min[qi], heap_mi[qi]);
    }
  }
}
#endif  // !MinGW Win32 permute-dot multi

#endif  // x86_64

}  // namespace

QueryPermuteDot build_permute_dot(std::span<const float> q_rot_row,
                                  std::span<const float> centroids, std::size_t dim) {
  assert(centroids.size() == 16);
  assert(q_rot_row.size() == dim);
  const std::size_t n_byte_groups = dim / 2;

  float cmax = 0.f;
  for (float c : centroids) cmax = std::max(cmax, std::abs(c));
  const float cs = (cmax > 0.f) ? (cmax / 127.f) : 1.f;
  QueryPermuteDot out;
  for (std::size_t i = 0; i < 16; ++i) {
    out.levels[i] = static_cast<std::int8_t>(
        std::clamp(std::round(centroids[i] / cs), -127.f, 127.f));
  }

  float qmax = 0.f;
  for (float v : q_rot_row) qmax = std::max(qmax, std::abs(v));
  float qs = qmax / 127.f;
  float inv_qs = 1.f;
  if (qs >= std::numeric_limits<float>::min()) {
    inv_qs = 1.f / qs;
  } else {
    qs = 1.f;
  }

  out.weights.assign(n_byte_groups * 2, 0);
  std::int32_t wsum = 0;
  for (std::size_t g = 0; g < n_byte_groups; ++g) {
    const std::size_t q4 = g / 4;
    const std::size_t j = g % 4;
    const auto lo = static_cast<std::int8_t>(
        std::clamp(std::round(q_rot_row[2 * g + 1] * inv_qs), -127.f, 127.f));
    const auto hi = static_cast<std::int8_t>(
        std::clamp(std::round(q_rot_row[2 * g] * inv_qs), -127.f, 127.f));
    out.weights[q4 * 8 + j] = lo;
    out.weights[q4 * 8 + 4 + j] = hi;
    wsum += static_cast<std::int32_t>(lo) + static_cast<std::int32_t>(hi);
  }

  out.zero = -128 * wsum;
  out.scale = cs * qs;
  out.bias = 0.f;
  return out;
}

std::vector<std::uint8_t> split_lut_for_vnni(std::span<const std::uint8_t> uint8_luts,
                                            std::size_t n_byte_groups) {
  assert(uint8_luts.size() == n_byte_groups * 32);
  assert(n_byte_groups % 4 == 0);
  std::vector<std::uint8_t> out(n_byte_groups * 32);
  for (std::size_t g0 = 0; g0 < n_byte_groups; g0 += 4) {
    const std::size_t c = (g0 / 4) * 128;
    for (std::size_t j = 0; j < 4; ++j) {
      const std::size_t src = (g0 + j) * 32;
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

void score_queries_vnni(const std::uint8_t* const* split_luts, const float* lut_scales,
                        const float* lut_biases, std::size_t nq,
                        std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t n_byte_groups,
                        std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                        float* const* heap_s, std::uint64_t* const* heap_i, std::size_t* heap_sz,
                        float* heap_min, std::size_t* heap_mi) {
#if defined(__x86_64__) || defined(_M_X64)
  if (nq == 1) {
    QueryLutView view{split_luts[0], lut_scales[0], lut_biases[0]};
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[0], heap_i[0], heap_sz[0], heap_min[0], heap_mi[0], 0.f);
    return;
  }
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  // MinGW Win64 miscompiles multi-query ZMM spills; use dual-block single path.
  for (std::size_t i = 0; i < nq; ++i) {
    QueryLutView view{split_luts[i], lut_scales[i], lut_biases[i]};
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[i], heap_i[i], heap_sz[i], heap_min[i], heap_mi[i], 0.f);
  }
#else
  score_queries_vnni_multi(split_luts, lut_scales, lut_biases, nq, blocked_codes, vec_scales,
                           n_byte_groups, n_vectors, n_blocks, k, heap_s, heap_i, heap_sz, heap_min,
                           heap_mi);
#endif
#else
  (void)split_luts;
  (void)lut_scales;
  (void)lut_biases;
  (void)nq;
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
  throw std::logic_error("score_queries_vnni is only available on x86_64");
#endif
}

void score_query_permute_dot(const QueryPermuteDot& pd, std::span<const std::uint8_t> blocked_codes,
                             std::span<const float> vec_scales, std::size_t n_byte_groups,
                             std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                             float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                             float& heap_min, std::size_t& heap_mi) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)pd;
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
  throw std::logic_error("score_query_permute_dot unavailable on MinGW Win64");
#else
  score_query_permute_dot_blk2(pd, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                               heap_s, heap_i, heap_sz, heap_min, heap_mi);
#endif
#else
  (void)pd;
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
  throw std::logic_error("score_query_permute_dot is only available on x86_64");
#endif
}

void score_queries_permute_dot(const QueryPermuteDot* const* pds, std::size_t nq,
                               std::span<const std::uint8_t> blocked_codes,
                               std::span<const float> vec_scales, std::size_t n_byte_groups,
                               std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                               float* const* heap_s, std::uint64_t* const* heap_i,
                               std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  // MinGW Win64: permute-dot kernels miscompile; degrade to classic VNNI via caller.
  (void)pds;
  (void)nq;
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
  throw std::logic_error("score_queries_permute_dot unavailable on MinGW Win64");
#else
  if (nq == 1) {
    score_query_permute_dot_blk2(*pds[0], blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                 n_blocks, k, heap_s[0], heap_i[0], heap_sz[0], heap_min[0],
                                 heap_mi[0]);
    return;
  }
  score_queries_permute_dot_multi(pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                  n_blocks, k, heap_s, heap_i, heap_sz, heap_min, heap_mi);
#endif
#else
  (void)pds;
  (void)nq;
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
  throw std::logic_error("score_queries_permute_dot is only available on x86_64");
#endif
}

}  // namespace vectorcache
