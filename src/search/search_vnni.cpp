#include "vectorcache/search/search_vnni.hpp"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/landmark_scan.hpp"
#include "vectorcache/search/topk_heap.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace vectorcache {
namespace {

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

/// turbovec `have_gfni`: GFNI high-nibble extract for permute-dot (~+4% MT).
bool have_gfni() {
  static std::once_flag once;
  static bool ok = false;
  std::call_once(once, [] {
    if (const char* v = std::getenv("TURBOVEC_NO_GFNI"); v && v[0] != '0') {
      ok = false;
      return;
    }
#if defined(_MSC_VER)
    int info[4] = {};
    __cpuid(info, 0);
    if (info[0] < 7) {
      ok = false;
      return;
    }
    __cpuidex(info, 7, 0);
    ok = (info[2] & (1 << 8)) != 0;  // GFNI
#else
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
      ok = false;
      return;
    }
    ok = (ecx & (1u << 8)) != 0;
#endif
  });
  return ok;
}

/// High nibbles: GFNI one-op (turbovec) or `srli`+and. Templated so the hot
/// loop is specialized (turbovec ships two `#[target_feature]` kernels).
template <bool kGfni>
inline __m512i hi_nibbles(__m512i c, __m512i m0f) {
  if constexpr (kGfni) {
    // Identity rows that map input bits 4..7 → output bits 0..3 (turbovec H38).
    return _mm512_gf2p8affine_epi64_epi8(
        c, _mm512_set1_epi64(static_cast<long long>(0x1020408000000000ull)), 0);
  } else {
    return _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
  }
}

/// turbovec `avx512_post_flush_heap_update`: early-out when no lane beats heap_min.
/// If `hit_bitmap` is non-null, only lanes with bit set are considered (pass-2 masked rerank).
VECTORCACHE_AVX512_FN
void flush_block_heap(__m512 f0, __m512 f1, std::size_t base_vec, std::size_t n_vectors,
                      const float* vec_scales, std::size_t k, float* heap_s, std::uint64_t* heap_i,
                      std::size_t& heap_sz, float& heap_min, std::size_t& heap_mi,
                      const std::uint64_t* id_map, float score_offset,
                      const std::uint64_t* hit_bitmap = nullptr, std::size_t vec_offset = 0,
                      std::size_t global_n_vectors = 0) {
  const std::size_t abs_base = base_vec + vec_offset;
  const std::size_t bm_n = global_n_vectors > 0 ? global_n_vectors : (vec_offset + n_vectors);
  const std::uint32_t bm =
      hit_bitmap ? block_hit_mask(hit_bitmap, abs_base, bm_n) : 0xFFFFFFFFu;
  if (bm == 0) {
    return;
  }
  const std::size_t end = std::min(base_vec + kBlock, n_vectors);
  const std::size_t end_lane = end - base_vec;
  const float* scales_ptr = vec_scales + base_vec;

  if (heap_sz >= k && end_lane == kBlock) {
    __m512 s0 = _mm512_mul_ps(f0, _mm512_loadu_ps(scales_ptr));
    __m512 s1 = _mm512_mul_ps(f1, _mm512_loadu_ps(scales_ptr + 16));
    if (score_offset != 0.f) {
      const __m512 off = _mm512_set1_ps(score_offset);
      s0 = _mm512_add_ps(s0, off);
      s1 = _mm512_add_ps(s1, off);
    }
    const __m512 thr = _mm512_set1_ps(heap_min);
    std::uint32_t m0 =
        static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s0, thr, _CMP_GT_OQ)) & (bm & 0xFFFFu);
    std::uint32_t m1 =
        static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s1, thr, _CMP_GT_OQ)) & (bm >> 16);
    if ((m0 | m1) == 0) {
      return;
    }
    alignas(64) float scores[kBlock];
    if (m0 != 0) {
      _mm512_storeu_ps(scores, s0);
    }
    if (m1 != 0) {
      _mm512_storeu_ps(scores + 16, s1);
    }
    for (std::size_t half = 0; half < 2; ++half) {
      std::uint32_t m = (half == 0) ? m0 : m1;
      while (m != 0) {
        const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
        m &= m - 1;
        const std::size_t lane = half * 16 + bit;
        const float score = scores[lane];
        if (score > heap_min) {
          heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k, score,
                               topk_map_id(id_map, base_vec + lane));
        }
      }
    }
    return;
  }

  // Sparse bitmap (pass-2): bit-scan only set lanes — avoid 32-lane scalar walk.
  alignas(64) float scores[kBlock];
  _mm512_storeu_ps(scores, f0);
  _mm512_storeu_ps(scores + 16, f1);
  std::uint32_t m = bm;
  if (end_lane < kBlock) {
    m &= (end_lane >= 32) ? 0xFFFFFFFFu : ((std::uint32_t{1} << end_lane) - 1u);
  }
  while (m != 0) {
    const unsigned lane = static_cast<unsigned>(std::countr_zero(m));
    m &= m - 1;
    heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k,
                         scores[lane] * scales_ptr[lane] + score_offset,
                         topk_map_id(id_map, base_vec + lane));
  }
}

/// SIMD early-out vs threshold (same shape as turbovec heap prune; our shortlist path).
/// If `hit_bitmap` is non-null, set bits for hits (pass-1 filter). If `out_scores` is non-null,
/// also append into the shortlist score/id buffers.
VECTORCACHE_AVX512_FN
bool flush_block_threshold(__m512 f0, __m512 f1, std::size_t base_vec, std::size_t n_vectors,
                           const float* vec_scales, float threshold, std::size_t capacity,
                           float* out_scores, std::uint64_t* out_ids, std::size_t* out_count,
                           const std::uint64_t* id_map, float score_offset,
                           std::uint64_t* hit_bitmap) {
  const bool record_list = out_scores != nullptr;
  const bool record_bits = hit_bitmap != nullptr;
  const bool record = record_list || record_bits;
  const std::size_t end = std::min(base_vec + kBlock, n_vectors);
  const std::size_t end_lane = end - base_vec;
  const float* scales_ptr = vec_scales + base_vec;

  auto record_hit = [&](std::size_t lane, float score) {
    const std::size_t vi = base_vec + lane;
    if (record_bits) {
      hit_bitmap[vi >> 6] |= (std::uint64_t{1} << (vi & 63u));
    }
    if (record_list) {
      out_scores[*out_count] = score;
      out_ids[*out_count] = topk_map_id(id_map, vi);
    }
    ++(*out_count);
  };

  if (end_lane == kBlock) {
    __m512 s0 = _mm512_mul_ps(f0, _mm512_loadu_ps(scales_ptr));
    __m512 s1 = _mm512_mul_ps(f1, _mm512_loadu_ps(scales_ptr + 16));
    if (score_offset != 0.f) {
      const __m512 off = _mm512_set1_ps(score_offset);
      s0 = _mm512_add_ps(s0, off);
      s1 = _mm512_add_ps(s1, off);
    }
    const __m512 thr = _mm512_set1_ps(threshold);
    const std::uint32_t m0 = static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s0, thr, _CMP_GE_OQ));
    const std::uint32_t m1 = static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s1, thr, _CMP_GE_OQ));
    // Empty-block early-out (turbovec heap prune). Dense hits: store+scalar
    // is faster than bit-scan — threshold shortlists are often dense.
    if ((m0 | m1) == 0) {
      return false;
    }
    alignas(64) float scores[kBlock];
    // Selective half-store (turbovec heap flush): skip empty halves.
    if (m0 != 0) {
      _mm512_storeu_ps(scores, s0);
    }
    if (m1 != 0) {
      _mm512_storeu_ps(scores + 16, s1);
    }
    for (std::size_t half = 0; half < 2; ++half) {
      std::uint32_t m = (half == 0) ? m0 : m1;
      while (m != 0) {
        if (record && *out_count >= capacity) {
          return true;
        }
        const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
        m &= m - 1;
        const std::size_t lane = half * 16 + bit;
        if (record) {
          record_hit(lane, scores[lane]);
        } else {
          ++(*out_count);
        }
      }
    }
    return record && *out_count >= capacity;
  }

  alignas(64) float scores[kBlock];
  _mm512_storeu_ps(scores, f0);
  _mm512_storeu_ps(scores + 16, f1);
  for (std::size_t lane = 0; lane < end_lane; ++lane) {
    if (record && *out_count >= capacity) {
      return true;
    }
    const float score = scores[lane] * scales_ptr[lane] + score_offset;
    if (score >= threshold) {
      if (record) {
        record_hit(lane, score);
      } else {
        ++(*out_count);
      }
    }
  }
  return record && *out_count >= capacity;
}

VECTORCACHE_AVX512_FN
void score_query_vnni_impl(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                           std::span<const float> vec_scales, std::size_t n_byte_groups,
                           std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                           std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                           std::size_t& heap_mi, float bias_corr, const std::uint64_t* id_map,
                           float score_offset) {
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
                       heap_min, heap_mi, id_map, score_offset);
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
                     heap_min, heap_mi, id_map, score_offset);
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
                              std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                              const std::uint64_t* id_map, const float* score_offsets) {
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= 10);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const __m512i kpos = _mm512_set1_epi32(static_cast<int>(0x30201000u));
  const __m512i ones = _mm512_set1_epi8(1);
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[10][2];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      acc[qi][0] = _mm512_setzero_si512();
      acc[qi][1] = _mm512_setzero_si512();
    }

    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        // turbovec search_multi_query_vnni::<false>: no software prefetch at NQ>1
        // (H5/H6 — MT regressions; hardware prefetcher owns the stream).
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
      const float off = score_offsets ? score_offsets[qi] : 0.f;
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s[qi], heap_i[qi],
                       heap_sz[qi], heap_min[qi], heap_mi[qi], id_map, off);
    }
  }
}
#endif  // !MinGW multi VNNI

#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
/// turbovec `permute_dot::<1, 8>`: eight interleaved code streams + depth-8 prefetch.
VECTORCACHE_AVX512_FN
template <bool kGfni>
void score_query_permute_dot_blk8(const QueryPermuteDot& pd,
                                  std::span<const std::uint8_t> blocked_codes,
                                  std::span<const float> vec_scales, std::size_t n_byte_groups,
                                  std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                  float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                  float& heap_min, std::size_t& heap_mi,
                                  const std::uint64_t* id_map, float score_offset,
                                  const std::uint64_t* hit_bitmap = nullptr,
                                  std::size_t vec_offset = 0, std::size_t global_n_vectors = 0) {
  assert(n_byte_groups % 4 == 0);
  constexpr std::size_t kBlk = 8;
  constexpr std::size_t kPfQuads = 8;
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const std::size_t bm_n = global_n_vectors > 0 ? global_n_vectors : (vec_offset + n_vectors);
  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pd.levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));
  const __m512 vs = _mm512_set1_ps(pd.scale);
  const __m512 vb = _mm512_set1_ps(pd.bias);
  const __m512i z = _mm512_set1_epi32(pd.zero);

  for (std::size_t b = 0; b < n_blocks; b += kBlk) {
    const std::size_t nb = std::min(kBlk, n_blocks - b);
    if (hit_bitmap) {
      bool any = false;
      for (std::size_t sub = 0; sub < nb; ++sub) {
        if (block_hit_mask(hit_bitmap, vec_offset + (b + sub) * kBlock, bm_n) != 0) {
          any = true;
          break;
        }
      }
      if (!any) {
        continue;
      }
    }
    __m512i acc[kBlk][2];
    for (std::size_t sub = 0; sub < nb; ++sub) {
      acc[sub][0] = z;
      acc[sub][1] = z;
    }
    const std::size_t group_base = b * block_bytes;
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        for (std::size_t sub = 0; sub < nb; ++sub) {
          const std::size_t block_base = group_base + sub * block_bytes;
          const std::size_t pf = block_base + (q4 + kPfQuads) * 128 + h * 64;
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
          const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c, m0f));
          const std::int8_t* wp = pd.weights.data() + q4 * 8;
          std::int32_t wlo_i = 0;
          std::int32_t whi_i = 0;
          std::memcpy(&wlo_i, wp, 4);
          std::memcpy(&whi_i, wp + 4, 4);
          acc[sub][h] = _mm512_dpbusd_epi32(acc[sub][h], vlo, _mm512_set1_epi32(wlo_i));
          acc[sub][h] = _mm512_dpbusd_epi32(acc[sub][h], vhi, _mm512_set1_epi32(whi_i));
        }
      }
    }
    for (std::size_t sub = 0; sub < nb; ++sub) {
      const std::size_t base_vec = (b + sub) * kBlock;
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[sub][0]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[sub][1]), vs, vb);
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                       heap_min, heap_mi, id_map, score_offset, hit_bitmap, vec_offset, bm_n);
    }
  }
}

/// turbovec `search_multi_query_permute_dot::<NQ,1>`: NQ is a compile-time
/// constant so the qi loop and `acc[NQ][2]` stay in registers (H34/H124).
/// Caller pads `pds`/`heap_*` to NQ; only the first `nq` lanes are flushed.
VECTORCACHE_AVX512_FN
template <bool kGfni, std::size_t NQ>
void score_queries_permute_dot_multi(const QueryPermuteDot* const* pds, std::size_t nq,
                                     std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                     float* const* heap_s, std::uint64_t* const* heap_i,
                                     std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                     const std::uint64_t* id_map, const float* score_offsets,
                                     const std::uint64_t* const* hit_bitmaps = nullptr,
                                     std::size_t vec_offset = 0, std::size_t global_n_vectors = 0) {
  static_assert(NQ == 8 || NQ == 10);
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= NQ);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(pds[0]->levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));

  const std::size_t bm_n = global_n_vectors > 0 ? global_n_vectors : (vec_offset + n_vectors);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    if (hit_bitmaps) {
      bool any = false;
      for (std::size_t qi = 0; qi < nq; ++qi) {
        if (hit_bitmaps[qi] &&
            block_hit_mask(hit_bitmaps[qi], vec_offset + base_vec, bm_n) != 0) {
          any = true;
          break;
        }
      }
      if (!any) {
        continue;
      }
    }
    const std::size_t block_base = b * block_bytes;
    __m512i acc[NQ][2];
    for (std::size_t qi = 0; qi < NQ; ++qi) {
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
        const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c, m0f));
        for (std::size_t qi = 0; qi < NQ; ++qi) {
          const std::int8_t* wp = pds[qi]->weights.data() + q4 * 8;
          // Unaligned i32 load → LLVM/MSVC fold into vpdpbusd {1to16} (turbovec).
          std::int32_t wlo_i;
          std::int32_t whi_i;
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
      const float off = score_offsets ? score_offsets[qi] : 0.f;
      const std::uint64_t* bm = hit_bitmaps ? hit_bitmaps[qi] : nullptr;
      flush_block_heap(f0, f1, base_vec, n_vectors, vec_scales.data(), k, heap_s[qi], heap_i[qi],
                       heap_sz[qi], heap_min[qi], heap_mi[qi], id_map, off, bm, vec_offset, bm_n);
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
                      std::size_t& heap_mi, float bias_corr, const std::uint64_t* id_map,
                      float score_offset) {
#if defined(__x86_64__) || defined(_M_X64)
  score_query_vnni_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                        heap_s, heap_i, heap_sz, heap_min, heap_mi, bias_corr, id_map,
                        score_offset);
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
  (void)id_map;
  (void)score_offset;
  throw std::logic_error("score_query_vnni is only available on x86_64");
#endif
}

void score_queries_vnni(const std::uint8_t* const* split_luts, const float* lut_scales,
                        const float* lut_biases, std::size_t nq,
                        std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t n_byte_groups,
                        std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                        float* const* heap_s, std::uint64_t* const* heap_i, std::size_t* heap_sz,
                        float* heap_min, std::size_t* heap_mi, const std::uint64_t* id_map,
                        const float* score_offsets) {
#if defined(__x86_64__) || defined(_M_X64)
  if (nq == 1) {
    QueryLutView view{split_luts[0], lut_scales[0], lut_biases[0]};
    const float off = score_offsets ? score_offsets[0] : 0.f;
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[0], heap_i[0], heap_sz[0], heap_min[0], heap_mi[0], 0.f, id_map,
                          off);
    return;
  }
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  // MinGW Win64 miscompiles multi-query ZMM spills; use dual-block single path.
  for (std::size_t i = 0; i < nq; ++i) {
    QueryLutView view{split_luts[i], lut_scales[i], lut_biases[i]};
    const float off = score_offsets ? score_offsets[i] : 0.f;
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[i], heap_i[i], heap_sz[i], heap_min[i], heap_mi[i], 0.f, id_map,
                          off);
  }
#else
  score_queries_vnni_multi(split_luts, lut_scales, lut_biases, nq, blocked_codes, vec_scales,
                           n_byte_groups, n_vectors, n_blocks, k, heap_s, heap_i, heap_sz, heap_min,
                           heap_mi, id_map, score_offsets);
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
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_vnni is only available on x86_64");
#endif
}

void score_query_permute_dot(const QueryPermuteDot& pd, std::span<const std::uint8_t> blocked_codes,
                             std::span<const float> vec_scales, std::size_t n_byte_groups,
                             std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                             float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                             float& heap_min, std::size_t& heap_mi, const std::uint64_t* id_map,
                             float score_offset, const std::uint64_t* hit_bitmap,
                             std::size_t vec_offset, std::size_t global_n_vectors) {
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
  (void)id_map;
  (void)score_offset;
  (void)hit_bitmap;
  (void)vec_offset;
  (void)global_n_vectors;
  throw std::logic_error("score_query_permute_dot unavailable on MinGW Win64");
#else
  if (have_gfni()) {
    score_query_permute_dot_blk8<true>(pd, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                       n_blocks, k, heap_s, heap_i, heap_sz, heap_min, heap_mi,
                                       id_map, score_offset, hit_bitmap, vec_offset,
                                       global_n_vectors);
  } else {
    score_query_permute_dot_blk8<false>(pd, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                        n_blocks, k, heap_s, heap_i, heap_sz, heap_min, heap_mi,
                                        id_map, score_offset, hit_bitmap, vec_offset,
                                        global_n_vectors);
  }
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
  (void)id_map;
  (void)score_offset;
  (void)hit_bitmap;
  (void)vec_offset;
  (void)global_n_vectors;
  throw std::logic_error("score_query_permute_dot is only available on x86_64");
#endif
}

void score_queries_permute_dot(const QueryPermuteDot* const* pds, std::size_t nq,
                               std::span<const std::uint8_t> blocked_codes,
                               std::span<const float> vec_scales, std::size_t n_byte_groups,
                               std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                               float* const* heap_s, std::uint64_t* const* heap_i,
                               std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                               const std::uint64_t* id_map, const float* score_offsets,
                               const std::uint64_t* const* hit_bitmaps, std::size_t vec_offset,
                               std::size_t global_n_vectors) {
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
  (void)id_map;
  (void)score_offsets;
  (void)hit_bitmaps;
  (void)vec_offset;
  (void)global_n_vectors;
  throw std::logic_error("score_queries_permute_dot unavailable on MinGW Win64");
#else
  if (nq == 1) {
    const float off = score_offsets ? score_offsets[0] : 0.f;
    const std::uint64_t* bm = hit_bitmaps ? hit_bitmaps[0] : nullptr;
    if (have_gfni()) {
      score_query_permute_dot_blk8<true>(*pds[0], blocked_codes, vec_scales, n_byte_groups,
                                         n_vectors, n_blocks, k, heap_s[0], heap_i[0], heap_sz[0],
                                         heap_min[0], heap_mi[0], id_map, off, bm, vec_offset,
                                         global_n_vectors);
    } else {
      score_query_permute_dot_blk8<false>(*pds[0], blocked_codes, vec_scales, n_byte_groups,
                                          n_vectors, n_blocks, k, heap_s[0], heap_i[0], heap_sz[0],
                                          heap_min[0], heap_mi[0], id_map, off, bm, vec_offset,
                                          global_n_vectors);
    }
    return;
  }
  // Pad to full NQ so the accumulate loop is compile-time constant (turbovec pad_qi).
  if (nq > 10) {
    throw std::invalid_argument("score_queries_permute_dot: nq > 10");
  }
  const std::size_t NQ = (nq > 8) ? 10 : 8;
  const QueryPermuteDot* pad_pds[10];
  float* pad_heap_s[10];
  std::uint64_t* pad_heap_i[10];
  std::size_t pad_heap_sz[10];
  float pad_heap_min[10];
  std::size_t pad_heap_mi[10];
  float pad_offs[10];
  const std::uint64_t* pad_bm[10];
  float dummy_s = 0.f;
  std::uint64_t dummy_i = 0;
  const std::size_t pad_qi = nq - 1;
  for (std::size_t i = 0; i < NQ; ++i) {
    const std::size_t src = (i < nq) ? i : pad_qi;
    pad_pds[i] = pds[src];
    if (i < nq) {
      pad_heap_s[i] = heap_s[i];
      pad_heap_i[i] = heap_i[i];
      pad_heap_sz[i] = heap_sz[i];
      pad_heap_min[i] = heap_min[i];
      pad_heap_mi[i] = heap_mi[i];
      pad_bm[i] = hit_bitmaps ? hit_bitmaps[i] : nullptr;
    } else {
      pad_heap_s[i] = &dummy_s;
      pad_heap_i[i] = &dummy_i;
      pad_heap_sz[i] = 0;
      pad_heap_min[i] = 0.f;
      pad_heap_mi[i] = 0;
      pad_bm[i] = nullptr;
    }
    pad_offs[i] = score_offsets ? score_offsets[src] : 0.f;
  }
  const float* offs_ptr = score_offsets ? pad_offs : nullptr;
  const std::uint64_t* const* bm_ptr = hit_bitmaps ? pad_bm : nullptr;
  if (NQ == 10) {
    if (have_gfni()) {
      score_queries_permute_dot_multi<true, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
          pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
          global_n_vectors);
    } else {
      score_queries_permute_dot_multi<false, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
          pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
          global_n_vectors);
    }
  } else if (have_gfni()) {
    score_queries_permute_dot_multi<true, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
        pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
        global_n_vectors);
  } else {
    score_queries_permute_dot_multi<false, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
        pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
        global_n_vectors);
  }
  for (std::size_t i = 0; i < nq; ++i) {
    heap_sz[i] = pad_heap_sz[i];
    heap_min[i] = pad_heap_min[i];
    heap_mi[i] = pad_heap_mi[i];
  }
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
  (void)id_map;
  (void)score_offsets;
  (void)hit_bitmaps;
  (void)vec_offset;
  (void)global_n_vectors;
  throw std::logic_error("score_queries_permute_dot is only available on x86_64");
#endif
}

#if defined(__x86_64__) || defined(_M_X64)
#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
namespace {

#if defined(__GNUC__) || defined(__clang__)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
#define VECTORCACHE_AVX512_FN_THR
#else
#define VECTORCACHE_AVX512_FN_THR __attribute__((noinline, noipa))
#endif
#else
#define VECTORCACHE_AVX512_FN_THR
#endif

/// Single-query threshold: BLK=8 streams + depth-8 prefetch (matches heap blk8 / turbovec).
VECTORCACHE_AVX512_FN_THR
template <bool kGfni>
void score_query_permute_dot_threshold_blk8(const QueryPermuteDot& pd,
                                            std::span<const std::uint8_t> blocked_codes,
                                            std::span<const float> vec_scales,
                                            std::size_t n_byte_groups, std::size_t n_vectors,
                                            std::size_t n_blocks, float threshold,
                                            std::size_t capacity, float* out_scores,
                                            std::uint64_t* out_ids, std::size_t* out_count,
                                            const std::uint64_t* id_map, float score_offset) {
  assert(n_byte_groups % 4 == 0);
  const bool count_only = out_scores == nullptr;
  if (!count_only && (capacity == 0 || *out_count >= capacity)) {
    return;
  }
  constexpr std::size_t kBlk = 8;
  constexpr std::size_t kPfQuads = 8;
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pd.levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));
  const __m512 vs = _mm512_set1_ps(pd.scale);
  const __m512 vb = _mm512_set1_ps(pd.bias);
  const __m512i z = _mm512_set1_epi32(pd.zero);

  for (std::size_t b = 0; b < n_blocks; b += kBlk) {
    if (!count_only && *out_count >= capacity) {
      return;
    }
    const std::size_t nb = std::min(kBlk, n_blocks - b);
    __m512i acc[kBlk][2];
    for (std::size_t sub = 0; sub < nb; ++sub) {
      acc[sub][0] = z;
      acc[sub][1] = z;
    }
    const std::size_t group_base = b * block_bytes;
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        for (std::size_t sub = 0; sub < nb; ++sub) {
          const std::size_t block_base = group_base + sub * block_bytes;
          const std::size_t pf = block_base + (q4 + kPfQuads) * 128 + h * 64;
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
          const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c, m0f));
          const std::int8_t* wp = pd.weights.data() + q4 * 8;
          std::int32_t wlo_i = 0;
          std::int32_t whi_i = 0;
          std::memcpy(&wlo_i, wp, 4);
          std::memcpy(&whi_i, wp + 4, 4);
          acc[sub][h] = _mm512_dpbusd_epi32(acc[sub][h], vlo, _mm512_set1_epi32(wlo_i));
          acc[sub][h] = _mm512_dpbusd_epi32(acc[sub][h], vhi, _mm512_set1_epi32(whi_i));
        }
      }
    }
    for (std::size_t sub = 0; sub < nb; ++sub) {
      const std::size_t base_vec = (b + sub) * kBlock;
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[sub][0]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[sub][1]), vs, vb);
      if (flush_block_threshold(f0, f1, base_vec, n_vectors, vec_scales.data(), threshold, capacity,
                                out_scores, out_ids, out_count, id_map, score_offset,
                                /*hit_bitmap=*/nullptr)) {
        return;
      }
    }
  }
}

/// Multi-query threshold: codes loaded once, BLK=1, depth-32 prefetch (turbovec NQ>1).
/// Const NQ + pad (turbovec H34): always accumulate all NQ lanes; flush first `nq`.
VECTORCACHE_AVX512_FN_THR
template <bool kGfni, std::size_t NQ>
void score_queries_permute_dot_threshold_multi(const QueryPermuteDot* const* pds, std::size_t nq,
                                               std::span<const std::uint8_t> blocked_codes,
                                               std::span<const float> vec_scales,
                                               std::size_t n_byte_groups, std::size_t n_vectors,
                                               std::size_t n_blocks, float threshold,
                                               std::size_t capacity, float* const* out_scores,
                                               std::uint64_t* const* out_ids,
                                               std::size_t* out_counts,
                                               const std::uint64_t* id_map,
                                               const float* score_offsets) {
  static_assert(NQ == 8 || NQ == 10);
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= NQ);
  const bool count_only = out_scores == nullptr;
  constexpr std::size_t kPfQuads = 32;
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(pds[0]->levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));

  auto all_full = [&]() {
    if (count_only) return false;
    for (std::size_t qi = 0; qi < nq; ++qi) {
      if (out_counts[qi] < capacity) return false;
    }
    return true;
  };

  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (all_full()) break;
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[NQ][2];
    for (std::size_t qi = 0; qi < NQ; ++qi) {
      const __m512i z = _mm512_set1_epi32(pds[qi]->zero);
      acc[qi][0] = z;
      acc[qi][1] = z;
    }
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const std::size_t pf = block_base + (q4 + kPfQuads) * 128 + h * 64;
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
        const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c, m0f));
        // Always score all NQ padded lanes (turbovec); capacity gated at block level.
        for (std::size_t qi = 0; qi < NQ; ++qi) {
          const std::int8_t* wp = pds[qi]->weights.data() + q4 * 8;
          std::int32_t wlo_i;
          std::int32_t whi_i;
          std::memcpy(&wlo_i, wp, 4);
          std::memcpy(&whi_i, wp + 4, 4);
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vlo, _mm512_set1_epi32(wlo_i));
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], vhi, _mm512_set1_epi32(whi_i));
        }
      }
    }
    for (std::size_t qi = 0; qi < nq; ++qi) {
      if (!count_only && out_counts[qi] >= capacity) continue;
      const __m512 vs = _mm512_set1_ps(pds[qi]->scale);
      const __m512 vb = _mm512_set1_ps(pds[qi]->bias);
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs, vb);
      const float off = score_offsets ? score_offsets[qi] : 0.f;
      float* scores_qi = count_only ? nullptr : out_scores[qi];
      std::uint64_t* ids_qi = count_only ? nullptr : out_ids[qi];
      flush_block_threshold(f0, f1, base_vec, n_vectors, vec_scales.data(), threshold, capacity,
                            scores_qi, ids_qi, &out_counts[qi], id_map, off,
                            /*hit_bitmap=*/nullptr);
    }
  }
}

VECTORCACHE_AVX512_FN_THR
void score_query_vnni_threshold_impl(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                     std::size_t capacity, float* out_scores,
                                     std::uint64_t* out_ids, std::size_t* out_count, float bias_corr,
                                     const std::uint64_t* id_map, float score_offset) {
  assert(n_byte_groups % 4 == 0);
  assert(lut.uint8_luts != nullptr);
  const bool count_only = out_scores == nullptr;
  if (!count_only && (capacity == 0 || *out_count >= capacity)) {
    return;
  }

  constexpr std::size_t kPfQuads = 8;
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
    if (!count_only && *out_count >= capacity) {
      return;
    }
    const std::size_t b = pb * 2;
    const std::size_t base0 = b * block_bytes;
    const std::size_t base1 = base0 + block_bytes;
    __m512i a0[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};
    __m512i a1[2] = {_mm512_setzero_si512(), _mm512_setzero_si512()};

    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        const std::size_t pf = base0 + (q4 + kPfQuads) * 128 + h * 64;
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
      if (flush_block_threshold(f0, f1, base_vec, n_vectors, vec_scales.data(), threshold, capacity,
                                out_scores, out_ids, out_count, id_map, score_offset,
                                /*hit_bitmap=*/nullptr)) {
        return;
      }
    }
  }

  for (std::size_t b = n_pairs * 2; b < n_blocks; ++b) {
    if (!count_only && *out_count >= capacity) {
      return;
    }
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
    if (flush_block_threshold(f0, f1, base_vec, n_vectors, vec_scales.data(), threshold, capacity,
                              out_scores, out_ids, out_count, id_map, score_offset,
                              /*hit_bitmap=*/nullptr)) {
      return;
    }
  }
}

VECTORCACHE_AVX512_FN_THR
void score_queries_vnni_threshold_multi(const std::uint8_t* const* split_luts,
                                        const float* lut_scales, const float* lut_biases,
                                        std::size_t nq, std::span<const std::uint8_t> blocked_codes,
                                        std::span<const float> vec_scales, std::size_t n_byte_groups,
                                        std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                        std::size_t capacity, float* const* out_scores,
                                        std::uint64_t* const* out_ids, std::size_t* out_counts,
                                        const std::uint64_t* id_map, const float* score_offsets) {
  assert(n_byte_groups % 4 == 0);
  assert(nq >= 1 && nq <= 10);
  const bool count_only = out_scores == nullptr;
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const __m512i kpos = _mm512_set1_epi32(static_cast<int>(0x30201000u));
  const __m512i ones = _mm512_set1_epi8(1);
  const std::size_t quads = n_byte_groups / 4;
  const std::size_t block_bytes = n_byte_groups * kBlock;

  auto all_full = [&]() {
    if (count_only) return false;
    for (std::size_t qi = 0; qi < nq; ++qi) {
      if (out_counts[qi] < capacity) return false;
    }
    return true;
  };

  for (std::size_t b = 0; b < n_blocks; ++b) {
    if (all_full()) break;
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[10][2];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      acc[qi][0] = _mm512_setzero_si512();
      acc[qi][1] = _mm512_setzero_si512();
    }
    for (std::size_t q4 = 0; q4 < quads; ++q4) {
      for (std::size_t h = 0; h < 2; ++h) {
        // turbovec multi VNNI: no software prefetch (NQ>1).
        const __m512i c = _mm512_loadu_si512(
            reinterpret_cast<const __m512i*>(blocked_codes.data() + block_base + q4 * 128 + h * 64));
        const __m512i ilo = _mm512_or_si512(_mm512_and_si512(c, m0f), kpos);
        const __m512i ihi =
            _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(c, 4), m0f), kpos);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          if (!count_only && out_counts[qi] >= capacity) continue;
          const std::uint8_t* tp = split_luts[qi] + q4 * 128;
          const __m512i tlo = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp));
          const __m512i thi = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(tp + 64));
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], _mm512_permutexvar_epi8(ilo, tlo), ones);
          acc[qi][h] = _mm512_dpbusd_epi32(acc[qi][h], _mm512_permutexvar_epi8(ihi, thi), ones);
        }
      }
    }
    for (std::size_t qi = 0; qi < nq; ++qi) {
      if (!count_only && out_counts[qi] >= capacity) continue;
      const __m512 vs = _mm512_set1_ps(lut_scales[qi]);
      const __m512 vb = _mm512_set1_ps(lut_biases[qi]);
      const __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs), vb);
      const __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs), vb);
      const float off = score_offsets ? score_offsets[qi] : 0.f;
      float* scores_qi = count_only ? nullptr : out_scores[qi];
      std::uint64_t* ids_qi = count_only ? nullptr : out_ids[qi];
      flush_block_threshold(f0, f1, base_vec, n_vectors, vec_scales.data(), threshold, capacity,
                            scores_qi, ids_qi, &out_counts[qi], id_map, off,
                            /*hit_bitmap=*/nullptr);
    }
  }
}

}  // namespace
#endif  // !MinGW
#endif  // x86_64

void score_query_permute_dot_threshold_fill(const QueryPermuteDot& pd,
                                            std::span<const std::uint8_t> blocked_codes,
                                            std::span<const float> vec_scales,
                                            std::size_t n_byte_groups, std::size_t n_vectors,
                                            std::size_t n_blocks, float threshold,
                                            std::size_t capacity, float* out_scores,
                                            std::uint64_t* out_ids, std::size_t* out_count,
                                            const std::uint64_t* id_map, float score_offset) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)pd;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_count;
  (void)id_map;
  (void)score_offset;
  throw std::logic_error("score_query_permute_dot_threshold_fill unavailable on MinGW Win64");
#else
  if (have_gfni()) {
    score_query_permute_dot_threshold_blk8<true>(
        pd, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold, capacity,
        out_scores, out_ids, out_count, id_map, score_offset);
  } else {
    score_query_permute_dot_threshold_blk8<false>(
        pd, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold, capacity,
        out_scores, out_ids, out_count, id_map, score_offset);
  }
#endif
#else
  (void)pd;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_count;
  (void)id_map;
  (void)score_offset;
  throw std::logic_error("score_query_permute_dot_threshold_fill requires x86_64");
#endif
}

void score_query_vnni_threshold_fill(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                     std::size_t capacity, float* out_scores,
                                     std::uint64_t* out_ids, std::size_t* out_count, float bias_corr,
                                     const std::uint64_t* id_map, float score_offset) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)lut;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_count;
  (void)bias_corr;
  (void)id_map;
  (void)score_offset;
  throw std::logic_error("score_query_vnni_threshold_fill unavailable on MinGW Win64");
#else
  score_query_vnni_threshold_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks,
                                  threshold, capacity, out_scores, out_ids, out_count, bias_corr,
                                  id_map, score_offset);
#endif
#else
  (void)lut;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_count;
  (void)bias_corr;
  (void)id_map;
  (void)score_offset;
  throw std::logic_error("score_query_vnni_threshold_fill requires x86_64");
#endif
}

void score_queries_permute_dot_threshold_fill(const QueryPermuteDot* const* pds, std::size_t nq,
                                              std::span<const std::uint8_t> blocked_codes,
                                              std::span<const float> vec_scales,
                                              std::size_t n_byte_groups, std::size_t n_vectors,
                                              std::size_t n_blocks, float threshold,
                                              std::size_t capacity, float* const* out_scores,
                                              std::uint64_t* const* out_ids, std::size_t* out_counts,
                                              const std::uint64_t* id_map,
                                              const float* score_offsets) {
  if (nq == 0) return;
  if (nq > 10) {
    throw std::invalid_argument("score_queries_permute_dot_threshold_fill: nq > 10");
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)pds;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_counts;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_permute_dot_threshold_fill unavailable on MinGW Win64");
#else
  if (nq == 1) {
    const float off = score_offsets ? score_offsets[0] : 0.f;
    float* scores0 = out_scores ? out_scores[0] : nullptr;
    std::uint64_t* ids0 = out_ids ? out_ids[0] : nullptr;
    if (have_gfni()) {
      score_query_permute_dot_threshold_blk8<true>(
          *pds[0], blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
          capacity, scores0, ids0, &out_counts[0], id_map, off);
    } else {
      score_query_permute_dot_threshold_blk8<false>(
          *pds[0], blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
          capacity, scores0, ids0, &out_counts[0], id_map, off);
    }
    return;
  }
  // Pad to full NQ (turbovec pad_qi / H34).
  const std::size_t NQ = (nq > 8) ? 10 : 8;
  const QueryPermuteDot* pad_pds[10];
  float* pad_scores[10];
  std::uint64_t* pad_ids[10];
  std::size_t pad_counts[10] = {};
  float pad_offs[10];
  float dummy_score = 0.f;
  std::uint64_t dummy_id = 0;
  const std::size_t pad_qi = nq - 1;
  for (std::size_t i = 0; i < NQ; ++i) {
    const std::size_t src = (i < nq) ? i : pad_qi;
    pad_pds[i] = pds[src];
    if (i < nq) {
      pad_scores[i] = out_scores ? out_scores[i] : nullptr;
      pad_ids[i] = out_ids ? out_ids[i] : nullptr;
      pad_counts[i] = out_counts[i];
    } else {
      pad_scores[i] = &dummy_score;
      pad_ids[i] = &dummy_id;
      pad_counts[i] = 0;
    }
    pad_offs[i] = score_offsets ? score_offsets[src] : 0.f;
  }
  float* const* scores_arg = out_scores ? pad_scores : nullptr;
  std::uint64_t* const* ids_arg = out_ids ? pad_ids : nullptr;
  const float* offs_ptr = score_offsets ? pad_offs : nullptr;
  if (NQ == 10) {
    if (have_gfni()) {
      score_queries_permute_dot_threshold_multi<true, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
          capacity, scores_arg, ids_arg, pad_counts, id_map, offs_ptr);
    } else {
      score_queries_permute_dot_threshold_multi<false, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
          capacity, scores_arg, ids_arg, pad_counts, id_map, offs_ptr);
    }
  } else if (have_gfni()) {
    score_queries_permute_dot_threshold_multi<true, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
        capacity, scores_arg, ids_arg, pad_counts, id_map, offs_ptr);
  } else {
    score_queries_permute_dot_threshold_multi<false, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
        capacity, scores_arg, ids_arg, pad_counts, id_map, offs_ptr);
  }
  for (std::size_t i = 0; i < nq; ++i) {
    out_counts[i] = pad_counts[i];
  }
#endif
#else
  (void)pds;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_counts;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_permute_dot_threshold_fill requires x86_64");
#endif
}

void score_queries_vnni_threshold_fill(const std::uint8_t* const* split_luts,
                                       const float* lut_scales, const float* lut_biases,
                                       std::size_t nq, std::span<const std::uint8_t> blocked_codes,
                                       std::span<const float> vec_scales, std::size_t n_byte_groups,
                                       std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                       std::size_t capacity, float* const* out_scores,
                                       std::uint64_t* const* out_ids, std::size_t* out_counts,
                                       const std::uint64_t* id_map, const float* score_offsets) {
  if (nq == 0) return;
  if (nq > 10) {
    throw std::invalid_argument("score_queries_vnni_threshold_fill: nq > 10");
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)split_luts;
  (void)lut_scales;
  (void)lut_biases;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_counts;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_vnni_threshold_fill unavailable on MinGW Win64");
#else
  if (nq == 1) {
    QueryLutView view{split_luts[0], lut_scales[0], lut_biases[0]};
    const float off = score_offsets ? score_offsets[0] : 0.f;
    float* scores0 = out_scores ? out_scores[0] : nullptr;
    std::uint64_t* ids0 = out_ids ? out_ids[0] : nullptr;
    score_query_vnni_threshold_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                    n_blocks, threshold, capacity, scores0, ids0, &out_counts[0],
                                    0.f, id_map, off);
    return;
  }
  score_queries_vnni_threshold_multi(split_luts, lut_scales, lut_biases, nq, blocked_codes,
                                     vec_scales, n_byte_groups, n_vectors, n_blocks, threshold,
                                     capacity, out_scores, out_ids, out_counts, id_map,
                                     score_offsets);
#endif
#else
  (void)split_luts;
  (void)lut_scales;
  (void)lut_biases;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)threshold;
  (void)capacity;
  (void)out_scores;
  (void)out_ids;
  (void)out_counts;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_vnni_threshold_fill requires x86_64");
#endif
}

}  // namespace vectorcache
