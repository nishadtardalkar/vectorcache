#include "vectorcache/search/search_vnni.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cassert>
#include <chrono>
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

std::atomic<VnniKernelProfile*> g_vnni_profile{nullptr};

inline void profile_add_ns(std::atomic<std::uint64_t>* bucket,
                           std::chrono::steady_clock::time_point t0,
                           std::chrono::steady_clock::time_point t1) {
  if (bucket == nullptr) {
    return;
  }
  const auto ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  if (ns > 0) {
    bucket->fetch_add(static_cast<std::uint64_t>(ns), std::memory_order_relaxed);
  }
}

inline VnniKernelProfile* active_vnni_profile() {
  return g_vnni_profile.load(std::memory_order_relaxed);
}

#if defined(__x86_64__) || defined(_M_X64)

// Keep multi-query kernels out-of-line. MinGW Win64 has a ZMM spill-alignment
// bug with some attributes; leave kernels unadorned there. GCC/Clang get
// turbovec-style per-fn ISA targets (VNNI / GFNI clones).
#if defined(__GNUC__) || defined(__clang__)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
#define VECTORCACHE_AVX512_FN
#else
// Per-fn ISA (turbovec #[target_feature]): VBMI is load-bearing for the
// classic VNNI LUT path — without it `vpermb` is emulated (~3× slower).
// VNNI+GFNI cover permute-dot / affine high-nibble extract.
#define VECTORCACHE_AVX512_FN                                                         \
  __attribute__((                                                                     \
      noinline, noipa,                                                                \
      target("avx512f,avx512bw,avx512vbmi,avx512vnni,gfni")))
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

/// Deinterleave 16 [α, r] pairs from two __m512 of AoS floats into SoA registers.
inline void load_scale_resid16(const float* sr, __m512& scales, __m512& resids) {
  const __m512 p0 = _mm512_loadu_ps(sr);
  const __m512 p1 = _mm512_loadu_ps(sr + 16);
  // idx bit4 selects p1; even/odd lanes pull α / r.
  const __m512i even =
      _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  const __m512i odd =
      _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  scales = _mm512_permutex2var_ps(p0, even, p1);
  resids = _mm512_permutex2var_ps(p0, odd, p1);
}

/// turbovec `avx512_post_flush_heap_update`: early-out when no lane beats heap_min.
/// If `hit_bitmap` is non-null, only lanes with bit set are considered (pass-2 masked rerank).
/// If `scale_resid` is non-null, α and ‖r‖ are AoS pairs at scale_resid[2*i]; else use
/// separate vec_scales / residual_norms streams.
VECTORCACHE_AVX512_FN
void flush_block_heap(__m512 f0, __m512 f1, std::size_t base_vec, std::size_t n_vectors,
                      const float* vec_scales, std::size_t k, float* heap_s, std::uint64_t* heap_i,
                      std::size_t& heap_sz, float& heap_min, std::size_t& heap_mi,
                      const std::uint64_t* id_map, float score_offset,
                      const std::uint64_t* hit_bitmap = nullptr, std::size_t vec_offset = 0,
                      std::size_t global_n_vectors = 0, const float* residual_norms = nullptr,
                      float residual_scale = 0.f, const float* scale_resid = nullptr) {
  if (base_vec >= n_vectors) {
    return;
  }
  const std::size_t abs_base = base_vec + vec_offset;
  const std::size_t bm_n = global_n_vectors > 0 ? global_n_vectors : (vec_offset + n_vectors);
  const std::uint32_t bm =
      hit_bitmap ? block_hit_mask(hit_bitmap, abs_base, bm_n) : 0xFFFFFFFFu;
  if (bm == 0) {
    return;
  }
  // One call scores kFlushChunk (32) lanes; 192-wide blocks invoke this 6×.
  const std::size_t end = std::min(base_vec + kFlushChunk, n_vectors);
  const std::size_t end_lane = end - base_vec;
  const bool use_aos = scale_resid != nullptr;
  const float* scales_ptr = use_aos ? nullptr : (vec_scales + base_vec);
  const float* resid_ptr =
      (!use_aos && residual_norms != nullptr && residual_scale != 0.f) ? (residual_norms + base_vec)
                                                                      : nullptr;
  const float* sr_ptr = use_aos ? (scale_resid + base_vec * 2) : nullptr;
  const bool apply_resid = use_aos ? (residual_scale != 0.f) : (resid_ptr != nullptr);

  auto corrected = [&](float lut_scaled, std::size_t lane) {
    float s = lut_scaled + score_offset;
    if (apply_resid) {
      const float r = use_aos ? sr_ptr[2 * lane + 1] : resid_ptr[lane];
      s += residual_scale * r;
    }
    return s;
  };
  auto scale_lane = [&](std::size_t lane) {
    return use_aos ? sr_ptr[2 * lane] : scales_ptr[lane];
  };

  // Full heap OR buffered collector (turbovec H99): threshold-prune whole chunks.
  // Collector size stays ~k/2 after compact, so we must not require heap_sz >= k.
  if ((heap_sz >= k || heap_mi == kHeapBuffered) && end_lane == kFlushChunk) {
    __m512 s0, s1;
    if (use_aos) {
      __m512 a0, r0, a1, r1;
      load_scale_resid16(sr_ptr, a0, r0);
      load_scale_resid16(sr_ptr + 32, a1, r1);
      s0 = _mm512_mul_ps(f0, a0);
      s1 = _mm512_mul_ps(f1, a1);
      if (score_offset != 0.f) {
        const __m512 off = _mm512_set1_ps(score_offset);
        s0 = _mm512_add_ps(s0, off);
        s1 = _mm512_add_ps(s1, off);
      }
      if (apply_resid) {
        const __m512 rs = _mm512_set1_ps(residual_scale);
        s0 = _mm512_fmadd_ps(rs, r0, s0);
        s1 = _mm512_fmadd_ps(rs, r1, s1);
      }
    } else {
      s0 = _mm512_mul_ps(f0, _mm512_loadu_ps(scales_ptr));
      s1 = _mm512_mul_ps(f1, _mm512_loadu_ps(scales_ptr + 16));
      if (score_offset != 0.f) {
        const __m512 off = _mm512_set1_ps(score_offset);
        s0 = _mm512_add_ps(s0, off);
        s1 = _mm512_add_ps(s1, off);
      }
      if (resid_ptr != nullptr) {
        const __m512 rs = _mm512_set1_ps(residual_scale);
        s0 = _mm512_fmadd_ps(rs, _mm512_loadu_ps(resid_ptr), s0);
        s1 = _mm512_fmadd_ps(rs, _mm512_loadu_ps(resid_ptr + 16), s1);
      }
    }
    const __m512 thr = _mm512_set1_ps(heap_min);
    std::uint32_t m0 =
        static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s0, thr, _CMP_GT_OQ)) & (bm & 0xFFFFu);
    std::uint32_t m1 =
        static_cast<std::uint32_t>(_mm512_cmp_ps_mask(s1, thr, _CMP_GT_OQ)) & (bm >> 16);
    if ((m0 | m1) == 0) {
      return;
    }
    alignas(64) float scores[kFlushChunk];
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
  alignas(64) float scores[kFlushChunk];
  _mm512_storeu_ps(scores, f0);
  _mm512_storeu_ps(scores + 16, f1);
  std::uint32_t m = bm;
  if (end_lane < kFlushChunk) {
    m &= (end_lane >= 32) ? 0xFFFFFFFFu : ((std::uint32_t{1} << end_lane) - 1u);
  }
  while (m != 0) {
    const unsigned lane = static_cast<unsigned>(std::countr_zero(m));
    m &= m - 1;
    heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k,
                         corrected(scores[lane] * scale_lane(lane), lane),
                         topk_map_id(id_map, base_vec + lane));
  }
}

/// Add 64 LUT bytes (lo+hi already summed as u8) into four u32 accumulators (16 lanes each).
VECTORCACHE_AVX512_FN
inline void add_u8x64_to_acc12(__m512i acc[12], std::size_t base_chunk, __m512i v8) {
  // MSVC requires imm8 lane index for extracti32x4.
  acc[base_chunk + 0] =
      _mm512_add_epi32(acc[base_chunk + 0], _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v8, 0)));
  acc[base_chunk + 1] =
      _mm512_add_epi32(acc[base_chunk + 1], _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v8, 1)));
  acc[base_chunk + 2] =
      _mm512_add_epi32(acc[base_chunk + 2], _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v8, 2)));
  acc[base_chunk + 3] =
      _mm512_add_epi32(acc[base_chunk + 3], _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v8, 3)));
}

/// 192×1 VNNI-LUT: one byte-group at a time; epilogue once per 192 vectors (6×32 flushes).
VECTORCACHE_AVX512_FN
void score_query_vnni_impl(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                           std::span<const float> vec_scales, std::size_t n_byte_groups,
                           std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                           std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                           std::size_t& heap_mi, float bias_corr, const std::uint64_t* id_map,
                           float score_offset, const float* residual_norms = nullptr,
                           float residual_scale = 0.f) {
  assert(lut.uint8_luts != nullptr);
  static_assert(kBlock == 192 && kChunksPerBlock == 6);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const float bias = lut.bias + bias_corr;
  const __m512 vs = _mm512_set1_ps(lut.scale);
  const __m512 vb = _mm512_set1_ps(bias);

  VnniKernelProfile* const prof = active_vnni_profile();
  std::atomic<std::uint64_t>* const vnni_ns = prof ? &prof->vnni_ns : nullptr;
  std::atomic<std::uint64_t>* const epi_ns = prof ? &prof->epilogue_ns : nullptr;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t block_base = b * block_bytes;
    const std::size_t base_vec = b * kBlock;
    __m512i acc[12];
    for (std::size_t i = 0; i < 12; ++i) acc[i] = _mm512_setzero_si512();

    const auto t_vnni0 = std::chrono::steady_clock::now();
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      // uint8_luts: half0 (hi nibble) at +0, half1 (lo nibble) at +16.
      const __m512i tbl_hi =
          _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(lut.uint8_luts + g * 32)));
      const __m512i tbl_lo =
          _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(lut.uint8_luts + g * 32 + 16)));
      const std::uint8_t* cp = blocked_codes.data() + block_base + g * kBlock;
      if (g + 1 < n_byte_groups) {
        const std::size_t pf = block_base + (g + 1) * kBlock;
        if (pf + 64 <= blocked_codes.size()) {
#if defined(_MSC_VER)
          _mm_prefetch(reinterpret_cast<const char*>(blocked_codes.data() + pf), _MM_HINT_T0);
#else
          __builtin_prefetch(blocked_codes.data() + pf, 0, 3);
#endif
        }
      }
      for (std::size_t s = 0; s < 3; ++s) {
        const __m512i c =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(cp + s * 64));
        const __m512i ilo = _mm512_and_si512(c, m0f);
        const __m512i ihi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
        const __m512i v =
            _mm512_add_epi8(_mm512_shuffle_epi8(tbl_lo, ilo), _mm512_shuffle_epi8(tbl_hi, ihi));
        add_u8x64_to_acc12(acc, s * 4, v);
      }
    }
    const auto t_vnni1 = std::chrono::steady_clock::now();
    profile_add_ns(vnni_ns, t_vnni0, t_vnni1);

    const auto t_epi0 = std::chrono::steady_clock::now();
    for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
      const std::size_t chunk_base = base_vec + c * kFlushChunk;
      if (chunk_base >= n_vectors) {
        break;
      }
      const __m512 f0 =
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[c * 2]), vs), vb);
      const __m512 f1 =
          _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[c * 2 + 1]), vs), vb);
      flush_block_heap(f0, f1, chunk_base, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                       heap_min, heap_mi, id_map, score_offset, nullptr, 0, 0, residual_norms,
                       residual_scale);
    }
    const auto t_epi1 = std::chrono::steady_clock::now();
    profile_add_ns(epi_ns, t_epi0, t_epi1);
  }
}

#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
/// 192×1 multi-query VNNI-LUT. Cap NQ at 2 (12 u32 ZMMs × NQ).
template <std::size_t NQ>
VECTORCACHE_AVX512_FN
void score_queries_vnni_multi(const std::uint8_t* const* split_luts, const float* lut_scales,
                              const float* lut_biases, std::size_t nq,
                              std::span<const std::uint8_t> blocked_codes,
                              std::span<const float> vec_scales, std::size_t n_byte_groups,
                              std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                              float* const* heap_s, std::uint64_t* const* heap_i,
                              std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                              const std::uint64_t* id_map, const float* score_offsets,
                              const float* residual_norms = nullptr,
                              const float* residual_scales = nullptr) {
  static_assert(NQ == 2);
  assert(nq >= 1 && nq <= NQ);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t block_bytes = n_byte_groups * kBlock;
  VnniKernelProfile* const prof = active_vnni_profile();
  std::atomic<std::uint64_t>* const vnni_ns = prof ? &prof->vnni_ns : nullptr;
  std::atomic<std::uint64_t>* const epi_ns = prof ? &prof->epilogue_ns : nullptr;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    __m512i acc[NQ][12];
    for (std::size_t qi = 0; qi < NQ; ++qi) {
      for (std::size_t i = 0; i < 12; ++i) acc[qi][i] = _mm512_setzero_si512();
    }

    const auto t_vnni0 = std::chrono::steady_clock::now();
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      __m512i tbl_hi[NQ];
      __m512i tbl_lo[NQ];
      for (std::size_t qi = 0; qi < NQ; ++qi) {
        tbl_hi[qi] = _mm512_broadcast_i32x4(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(split_luts[qi] + g * 32)));
        tbl_lo[qi] = _mm512_broadcast_i32x4(_mm_loadu_si128(
            reinterpret_cast<const __m128i*>(split_luts[qi] + g * 32 + 16)));
      }
      const std::uint8_t* cp = blocked_codes.data() + block_base + g * kBlock;
      for (std::size_t s = 0; s < 3; ++s) {
        const __m512i c =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(cp + s * 64));
        const __m512i ilo = _mm512_and_si512(c, m0f);
        const __m512i ihi = _mm512_and_si512(_mm512_srli_epi16(c, 4), m0f);
        for (std::size_t qi = 0; qi < NQ; ++qi) {
          const __m512i v = _mm512_add_epi8(_mm512_shuffle_epi8(tbl_lo[qi], ilo),
                                            _mm512_shuffle_epi8(tbl_hi[qi], ihi));
          add_u8x64_to_acc12(acc[qi], s * 4, v);
        }
      }
    }
    const auto t_vnni1 = std::chrono::steady_clock::now();
    profile_add_ns(vnni_ns, t_vnni0, t_vnni1);

    const auto t_epi0 = std::chrono::steady_clock::now();
    for (std::size_t qi = 0; qi < nq; ++qi) {
      const __m512 vs = _mm512_set1_ps(lut_scales[qi]);
      const __m512 vb = _mm512_set1_ps(lut_biases[qi]);
      const float off = score_offsets ? score_offsets[qi] : 0.f;
      const float rs = residual_scales ? residual_scales[qi] : 0.f;
      for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
        const std::size_t chunk_base = base_vec + c * kFlushChunk;
        if (chunk_base >= n_vectors) {
          break;
        }
        const __m512 f0 =
            _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][c * 2]), vs), vb);
        const __m512 f1 =
            _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][c * 2 + 1]), vs), vb);
        flush_block_heap(f0, f1, chunk_base, n_vectors, vec_scales.data(), k, heap_s[qi],
                         heap_i[qi], heap_sz[qi], heap_min[qi], heap_mi[qi], id_map, off, nullptr,
                         0, 0, residual_norms, rs);
      }
    }
    const auto t_epi1 = std::chrono::steady_clock::now();
    profile_add_ns(epi_ns, t_epi0, t_epi1);
  }
}

/// Add `w * level_u8[0..15]` into one u32 accumulator (levels still xor 0x80).
inline void madd_level_w16(__m512i& acc, __m128i level_u8, std::int32_t w) {
  acc = _mm512_add_epi32(
      acc, _mm512_mullo_epi32(_mm512_cvtepu8_epi32(level_u8), _mm512_set1_epi32(w)));
}

/// 192×1 VNNI → dense α·lut rows (no heap). Chunk-outer so NQ≤10 keeps 2 ZMMs/query.
VECTORCACHE_AVX512_FN
void score_queries_vnni_to_rows_impl(const std::uint8_t* const* split_luts, const float* lut_scales,
                                     const float* lut_biases, std::size_t nq,
                                     std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks,
                                     float* const* score_rows) {
  assert(nq >= 1 && nq <= 10);
  static_assert(kBlock == 192 && kChunksPerBlock == 6);
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t block_bytes = n_byte_groups * kBlock;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t block_base = b * block_bytes;
    for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
      const std::size_t chunk_base = base_vec + c * kFlushChunk;
      if (chunk_base >= n_vectors) {
        break;
      }
      __m512i acc[10][2];
      for (std::size_t qi = 0; qi < nq; ++qi) {
        acc[qi][0] = _mm512_setzero_si512();
        acc[qi][1] = _mm512_setzero_si512();
      }
      for (std::size_t g = 0; g < n_byte_groups; ++g) {
        __m512i tbl_hi[10];
        __m512i tbl_lo[10];
        for (std::size_t qi = 0; qi < nq; ++qi) {
          tbl_hi[qi] = _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(split_luts[qi] + g * 32)));
          tbl_lo[qi] = _mm512_broadcast_i32x4(_mm_loadu_si128(
              reinterpret_cast<const __m128i*>(split_luts[qi] + g * 32 + 16)));
        }
        const std::uint8_t* cp = blocked_codes.data() + block_base + g * kBlock + c * kFlushChunk;
        const __m256i c32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp));
        const __m512i c512 = _mm512_castsi256_si512(c32);
        const __m512i ilo = _mm512_and_si512(c512, m0f);
        const __m512i ihi = _mm512_and_si512(_mm512_srli_epi16(c512, 4), m0f);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          const __m512i vlo = _mm512_shuffle_epi8(tbl_lo[qi], ilo);
          const __m512i vhi = _mm512_shuffle_epi8(tbl_hi[qi], ihi);
          const __m512i v = _mm512_add_epi8(vlo, vhi);
          acc[qi][0] = _mm512_add_epi32(
              acc[qi][0], _mm512_cvtepu8_epi32(_mm512_castsi512_si128(v)));
          acc[qi][1] = _mm512_add_epi32(
              acc[qi][1],
              _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(v, 1)));
        }
      }
      const float* scales_ptr = vec_scales.data() + chunk_base;
      const std::size_t chunk_n = std::min(kFlushChunk, n_vectors - chunk_base);
      for (std::size_t qi = 0; qi < nq; ++qi) {
        const __m512 vs = _mm512_set1_ps(lut_scales[qi]);
        const __m512 vb = _mm512_set1_ps(lut_biases[qi]);
        __m512 f0 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs), vb);
        __m512 f1 = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs), vb);
        f0 = _mm512_mul_ps(f0, _mm512_loadu_ps(scales_ptr));
        f1 = _mm512_mul_ps(f1, _mm512_loadu_ps(scales_ptr + 16));
        float* out = score_rows[qi] + chunk_base;
        if (chunk_n == kFlushChunk) {
          _mm512_storeu_ps(out, f0);
          _mm512_storeu_ps(out + 16, f1);
        } else {
          alignas(64) float tmp[kFlushChunk];
          _mm512_storeu_ps(tmp, f0);
          _mm512_storeu_ps(tmp + 16, f1);
          std::memcpy(out, tmp, chunk_n * sizeof(float));
        }
      }
    }
  }
}
#endif  // !MinGW multi VNNI

#if !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
/// 192×1 permute-dot: group-outer, 12×u32 accums, flush 6×32.
template <bool kGfni>
VECTORCACHE_AVX512_FN
void score_query_permute_dot_blk8(const QueryPermuteDot& pd,
                                  std::span<const std::uint8_t> blocked_codes,
                                  std::span<const float> vec_scales, std::size_t n_byte_groups,
                                  std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                  float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                  float& heap_min, std::size_t& heap_mi,
                                  const std::uint64_t* id_map, float score_offset,
                                  const std::uint64_t* hit_bitmap = nullptr,
                                  std::size_t vec_offset = 0, std::size_t global_n_vectors = 0,
                                  const float* residual_norms = nullptr,
                                  float residual_scale = 0.f, const float* scale_resid = nullptr) {
  static_assert(kBlock == 192 && kChunksPerBlock == 6);
  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::size_t block_bytes = n_byte_groups * kBlock;
  const std::size_t bm_n = global_n_vectors > 0 ? global_n_vectors : (vec_offset + n_vectors);
  const __m512i levels = _mm512_xor_si512(
      _mm512_broadcast_i32x4(_mm_loadu_si128(reinterpret_cast<const __m128i*>(pd.levels.data()))),
      _mm512_set1_epi8(static_cast<char>(0x80)));
  const __m512 vs = _mm512_set1_ps(pd.scale);
  const __m512 vb = _mm512_set1_ps(pd.bias);
  const __m512i z = _mm512_set1_epi32(pd.zero);

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    if (hit_bitmap && !block_any_hit(hit_bitmap, vec_offset + base_vec, bm_n)) {
      continue;
    }
    const std::size_t block_base = b * block_bytes;
    __m512i acc[12];
    for (std::size_t i = 0; i < 12; ++i) {
      acc[i] = z;
    }
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      const std::int32_t wlo = static_cast<std::int32_t>(pd.weights[2 * g]);
      const std::int32_t whi = static_cast<std::int32_t>(pd.weights[2 * g + 1]);
      const std::uint8_t* cp = blocked_codes.data() + block_base + g * kBlock;
      if (g + 1 < n_byte_groups) {
        const std::size_t pf = block_base + (g + 1) * kBlock;
        if (pf + 64 <= blocked_codes.size()) {
#if defined(_MSC_VER)
          _mm_prefetch(reinterpret_cast<const char*>(blocked_codes.data() + pf), _MM_HINT_T0);
#else
          __builtin_prefetch(blocked_codes.data() + pf, 0, 3);
#endif
        }
      }
      for (std::size_t s = 0; s < 3; ++s) {
        const __m512i c =
            _mm512_loadu_si512(reinterpret_cast<const __m512i*>(cp + s * 64));
        const __m512i vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c, m0f));
        const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c, m0f));
        const std::size_t a0 = s * 4;
        madd_level_w16(acc[a0 + 0], _mm512_extracti32x4_epi32(vlo, 0), wlo);
        madd_level_w16(acc[a0 + 1], _mm512_extracti32x4_epi32(vlo, 1), wlo);
        madd_level_w16(acc[a0 + 2], _mm512_extracti32x4_epi32(vlo, 2), wlo);
        madd_level_w16(acc[a0 + 3], _mm512_extracti32x4_epi32(vlo, 3), wlo);
        madd_level_w16(acc[a0 + 0], _mm512_extracti32x4_epi32(vhi, 0), whi);
        madd_level_w16(acc[a0 + 1], _mm512_extracti32x4_epi32(vhi, 1), whi);
        madd_level_w16(acc[a0 + 2], _mm512_extracti32x4_epi32(vhi, 2), whi);
        madd_level_w16(acc[a0 + 3], _mm512_extracti32x4_epi32(vhi, 3), whi);
      }
    }
    for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
      const std::size_t chunk_base = base_vec + c * kFlushChunk;
      if (chunk_base >= n_vectors) {
        break;
      }
      const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c * 2]), vs, vb);
      const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[c * 2 + 1]), vs, vb);
      flush_block_heap(f0, f1, chunk_base, n_vectors, vec_scales.data(), k, heap_s, heap_i, heap_sz,
                       heap_min, heap_mi, id_map, score_offset, hit_bitmap, vec_offset, bm_n,
                       residual_norms, residual_scale, scale_resid);
    }
  }
}

/// Multi-query 192×1 permute-dot: chunk-outer keeps `acc[NQ][2]` in registers.
template <bool kGfni, std::size_t NQ>
VECTORCACHE_AVX512_FN
void score_queries_permute_dot_multi(const QueryPermuteDot* const* pds, std::size_t nq,
                                     std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                     float* const* heap_s, std::uint64_t* const* heap_i,
                                     std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                     const std::uint64_t* id_map, const float* score_offsets,
                                     const std::uint64_t* const* hit_bitmaps = nullptr,
                                     std::size_t vec_offset = 0, std::size_t global_n_vectors = 0,
                                     const float* residual_norms = nullptr,
                                     const float* residual_scales = nullptr,
                                     const float* scale_resid = nullptr) {
  static_assert(NQ == 8 || NQ == 10);
  assert(nq >= 1 && nq <= NQ);
  static_assert(kBlock == 192 && kChunksPerBlock == 6);

  const __m512i m0f = _mm512_set1_epi8(static_cast<char>(0x0F));
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
        if (hit_bitmaps[qi] && block_any_hit(hit_bitmaps[qi], vec_offset + base_vec, bm_n)) {
          any = true;
          break;
        }
      }
      if (!any) {
        continue;
      }
    }
    const std::size_t block_base = b * block_bytes;
    for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
      const std::size_t chunk_base = base_vec + c * kFlushChunk;
      if (chunk_base >= n_vectors) {
        break;
      }
      __m512i acc[NQ][2];
      for (std::size_t qi = 0; qi < NQ; ++qi) {
        const __m512i z = _mm512_set1_epi32(pds[qi]->zero);
        acc[qi][0] = z;
        acc[qi][1] = z;
      }
      for (std::size_t g = 0; g < n_byte_groups; ++g) {
        std::int32_t wlo[NQ];
        std::int32_t whi[NQ];
        for (std::size_t qi = 0; qi < NQ; ++qi) {
          wlo[qi] = static_cast<std::int32_t>(pds[qi]->weights[2 * g]);
          whi[qi] = static_cast<std::int32_t>(pds[qi]->weights[2 * g + 1]);
        }
        const std::uint8_t* cp = blocked_codes.data() + block_base + g * kBlock + c * kFlushChunk;
        const __m256i c32 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp));
        const __m512i c512 = _mm512_castsi256_si512(c32);
        const __m512i vlo = _mm512_shuffle_epi8(levels, _mm512_and_si512(c512, m0f));
        const __m512i vhi = _mm512_shuffle_epi8(levels, hi_nibbles<kGfni>(c512, m0f));
        for (std::size_t qi = 0; qi < NQ; ++qi) {
          madd_level_w16(acc[qi][0], _mm512_castsi512_si128(vlo), wlo[qi]);
          madd_level_w16(acc[qi][1], _mm512_extracti32x4_epi32(vlo, 1), wlo[qi]);
          madd_level_w16(acc[qi][0], _mm512_castsi512_si128(vhi), whi[qi]);
          madd_level_w16(acc[qi][1], _mm512_extracti32x4_epi32(vhi, 1), whi[qi]);
        }
      }
      for (std::size_t qi = 0; qi < nq; ++qi) {
        const __m512 vs = _mm512_set1_ps(pds[qi]->scale);
        const __m512 vb = _mm512_set1_ps(pds[qi]->bias);
        const __m512 f0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][0]), vs, vb);
        const __m512 f1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc[qi][1]), vs, vb);
        const float off = score_offsets ? score_offsets[qi] : 0.f;
        const float rs = residual_scales ? residual_scales[qi] : 0.f;
        const std::uint64_t* bm = hit_bitmaps ? hit_bitmaps[qi] : nullptr;
        flush_block_heap(f0, f1, chunk_base, n_vectors, vec_scales.data(), k, heap_s[qi],
                         heap_i[qi], heap_sz[qi], heap_min[qi], heap_mi[qi], id_map, off, bm,
                         vec_offset, bm_n, residual_norms, rs, scale_resid);
      }
    }
  }
}

#endif  // !MinGW Win32 permute-dot kernels

#endif  // x86_64

}  // namespace

void vnni_profile_begin(VnniKernelProfile* profile) {
  g_vnni_profile.store(profile, std::memory_order_release);
}

void vnni_profile_end() {
  g_vnni_profile.store(nullptr, std::memory_order_release);
}

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

  // 192×1: per group [lo_w, hi_w] (no quad packing).
  out.weights.assign(n_byte_groups * 2, 0);
  std::int32_t wsum = 0;
  for (std::size_t g = 0; g < n_byte_groups; ++g) {
    const auto lo = static_cast<std::int8_t>(
        std::clamp(std::round(q_rot_row[2 * g + 1] * inv_qs), -127.f, 127.f));
    const auto hi = static_cast<std::int8_t>(
        std::clamp(std::round(q_rot_row[2 * g] * inv_qs), -127.f, 127.f));
    out.weights[2 * g] = lo;
    out.weights[2 * g + 1] = hi;
    wsum += static_cast<std::int32_t>(lo) + static_cast<std::int32_t>(hi);
  }

  out.zero = -128 * wsum;
  out.scale = cs * qs;
  out.bias = 0.f;
  return out;
}

std::vector<std::uint8_t> split_lut_for_vnni(std::span<const std::uint8_t> uint8_luts,
                                            std::size_t n_byte_groups) {
  // 192×1: keep per-group [hi16|lo16] layout (no 32×4 quad reshape).
  assert(uint8_luts.size() == n_byte_groups * 32);
  (void)n_byte_groups;
  return std::vector<std::uint8_t>(uint8_luts.begin(), uint8_luts.end());
}

void score_query_vnni(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                      std::span<const float> vec_scales, std::size_t n_byte_groups,
                      std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                      std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                      std::size_t& heap_mi, float bias_corr, const std::uint64_t* id_map,
                      float score_offset, const float* residual_norms, float residual_scale) {
#if defined(__x86_64__) || defined(_M_X64)
  score_query_vnni_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                        heap_s, heap_i, heap_sz, heap_min, heap_mi, bias_corr, id_map,
                        score_offset, residual_norms, residual_scale);
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
  (void)residual_norms;
  (void)residual_scale;
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
                        const float* score_offsets, const float* residual_norms,
                        const float* residual_scales) {
#if defined(__x86_64__) || defined(_M_X64)
  if (nq == 1) {
    QueryLutView view{split_luts[0], lut_scales[0], lut_biases[0]};
    const float off = score_offsets ? score_offsets[0] : 0.f;
    const float rs = residual_scales ? residual_scales[0] : 0.f;
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[0], heap_i[0], heap_sz[0], heap_min[0], heap_mi[0], 0.f, id_map,
                          off, residual_norms, rs);
    return;
  }
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  // MinGW Win64 miscompiles multi-query ZMM spills; use dual-block single path.
  for (std::size_t i = 0; i < nq; ++i) {
    QueryLutView view{split_luts[i], lut_scales[i], lut_biases[i]};
    const float off = score_offsets ? score_offsets[i] : 0.f;
    const float rs = residual_scales ? residual_scales[i] : 0.f;
    score_query_vnni_impl(view, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                          heap_s[i], heap_i[i], heap_sz[i], heap_min[i], heap_mi[i], 0.f, id_map,
                          off, residual_norms, rs);
  }
#else
  // 192×1: fused batch width 2; larger nq is multiple passes.
  if (nq > 2) {
    throw std::invalid_argument("score_queries_vnni: nq > 2 (caller must batch)");
  }
  const std::uint8_t* pad_luts[2];
  float pad_scales[2];
  float pad_biases[2];
  float* pad_heap_s[2];
  std::uint64_t* pad_heap_i[2];
  std::size_t pad_heap_sz[2];
  float pad_heap_min[2];
  std::size_t pad_heap_mi[2];
  float pad_offs[2];
  float pad_rs[2];
  float dummy_s = 0.f;
  std::uint64_t dummy_i = 0;
  const std::size_t pad_qi = nq - 1;
  for (std::size_t i = 0; i < 2; ++i) {
    const std::size_t src = (i < nq) ? i : pad_qi;
    pad_luts[i] = split_luts[src];
    pad_scales[i] = lut_scales[src];
    pad_biases[i] = lut_biases[src];
    pad_heap_s[i] = (i < nq) ? heap_s[i] : &dummy_s;
    pad_heap_i[i] = (i < nq) ? heap_i[i] : &dummy_i;
    pad_heap_sz[i] = (i < nq) ? heap_sz[i] : 0;
    pad_heap_min[i] = (i < nq) ? heap_min[i] : 0.f;
    pad_heap_mi[i] = (i < nq) ? heap_mi[i] : 0;
    pad_offs[i] = score_offsets ? score_offsets[src] : 0.f;
    pad_rs[i] = residual_scales ? residual_scales[src] : 0.f;
  }
  const float* offs_ptr = score_offsets ? pad_offs : nullptr;
  const float* rs_ptr = residual_scales ? pad_rs : nullptr;
  score_queries_vnni_multi<2>(pad_luts, pad_scales, pad_biases, nq, blocked_codes, vec_scales,
                              n_byte_groups, n_vectors, n_blocks, k, pad_heap_s, pad_heap_i,
                              pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr,
                              residual_norms, rs_ptr);
  for (std::size_t i = 0; i < nq; ++i) {
    heap_sz[i] = pad_heap_sz[i];
    heap_min[i] = pad_heap_min[i];
    heap_mi[i] = pad_heap_mi[i];
  }
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
  (void)residual_norms;
  (void)residual_scales;
  throw std::logic_error("score_queries_vnni is only available on x86_64");
#endif
}

void score_queries_vnni_to_rows(const std::uint8_t* const* split_luts, const float* lut_scales,
                                const float* lut_biases, std::size_t nq,
                                std::span<const std::uint8_t> blocked_codes,
                                std::span<const float> vec_scales, std::size_t n_byte_groups,
                                std::size_t n_vectors, std::size_t n_blocks,
                                float* const* score_rows) {
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  (void)split_luts;
  (void)lut_scales;
  (void)lut_biases;
  (void)nq;
  (void)blocked_codes;
  (void)vec_scales;
  (void)n_byte_groups;
  (void)n_vectors;
  (void)n_blocks;
  (void)score_rows;
  throw std::logic_error("score_queries_vnni_to_rows unavailable on MinGW Win64");
#else
  if (nq == 0 || nq > 10) {
    throw std::invalid_argument("score_queries_vnni_to_rows: nq must be in 1..10");
  }
  score_queries_vnni_to_rows_impl(split_luts, lut_scales, lut_biases, nq, blocked_codes, vec_scales,
                                  n_byte_groups, n_vectors, n_blocks, score_rows);
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
  (void)score_rows;
  throw std::logic_error("score_queries_vnni_to_rows is only available on x86_64");
#endif
}

void score_query_permute_dot(const QueryPermuteDot& pd, std::span<const std::uint8_t> blocked_codes,
                             std::span<const float> vec_scales, std::size_t n_byte_groups,
                             std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                             float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                             float& heap_min, std::size_t& heap_mi, const std::uint64_t* id_map,
                             float score_offset, const std::uint64_t* hit_bitmap,
                             std::size_t vec_offset, std::size_t global_n_vectors,
                             const float* residual_norms, float residual_scale,
                             const float* scale_resid) {
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
  (void)residual_norms;
  (void)residual_scale;
  (void)scale_resid;
  throw std::logic_error("score_query_permute_dot unavailable on MinGW Win64");
#else
  if (have_gfni()) {
    score_query_permute_dot_blk8<true>(pd, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                       n_blocks, k, heap_s, heap_i, heap_sz, heap_min, heap_mi,
                                       id_map, score_offset, hit_bitmap, vec_offset,
                                       global_n_vectors, residual_norms, residual_scale, scale_resid);
  } else {
    score_query_permute_dot_blk8<false>(
        pd, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, heap_s, heap_i,
        heap_sz, heap_min, heap_mi, id_map, score_offset, hit_bitmap, vec_offset, global_n_vectors,
        residual_norms, residual_scale, scale_resid);
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
  (void)residual_norms;
  (void)residual_scale;
  (void)scale_resid;
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
                               std::size_t global_n_vectors, const float* residual_norms,
                               const float* residual_scales, const float* scale_resid) {
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
  (void)residual_norms;
  (void)residual_scales;
  (void)scale_resid;
  throw std::logic_error("score_queries_permute_dot unavailable on MinGW Win64");
#else
  if (nq == 1) {
    const float off = score_offsets ? score_offsets[0] : 0.f;
    const float rs = residual_scales ? residual_scales[0] : 0.f;
    const std::uint64_t* bm = hit_bitmaps ? hit_bitmaps[0] : nullptr;
    if (have_gfni()) {
      score_query_permute_dot_blk8<true>(
          *pds[0], blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, heap_s[0],
          heap_i[0], heap_sz[0], heap_min[0], heap_mi[0], id_map, off, bm, vec_offset,
          global_n_vectors, residual_norms, rs, scale_resid);
    } else {
      score_query_permute_dot_blk8<false>(
          *pds[0], blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, heap_s[0],
          heap_i[0], heap_sz[0], heap_min[0], heap_mi[0], id_map, off, bm, vec_offset,
          global_n_vectors, residual_norms, rs, scale_resid);
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
  float pad_rs[10];
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
      pad_heap_min[i] = -std::numeric_limits<float>::infinity();
      pad_heap_mi[i] = 0;
      pad_bm[i] = nullptr;
    }
    pad_offs[i] = score_offsets ? score_offsets[src] : 0.f;
    pad_rs[i] = residual_scales ? residual_scales[src] : 0.f;
  }
  const float* offs_ptr = score_offsets ? pad_offs : nullptr;
  const float* rs_ptr = residual_scales ? pad_rs : nullptr;
  const std::uint64_t* const* bm_ptr = hit_bitmaps ? pad_bm : nullptr;
  if (NQ == 10) {
    if (have_gfni()) {
      score_queries_permute_dot_multi<true, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
          pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
          global_n_vectors, residual_norms, rs_ptr, scale_resid);
    } else {
      score_queries_permute_dot_multi<false, 10>(
          pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
          pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
          global_n_vectors, residual_norms, rs_ptr, scale_resid);
    }
  } else if (have_gfni()) {
    score_queries_permute_dot_multi<true, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
        pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
        global_n_vectors, residual_norms, rs_ptr, scale_resid);
  } else {
    score_queries_permute_dot_multi<false, 8>(
        pad_pds, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k, pad_heap_s,
        pad_heap_i, pad_heap_sz, pad_heap_min, pad_heap_mi, id_map, offs_ptr, bm_ptr, vec_offset,
        global_n_vectors, residual_norms, rs_ptr, scale_resid);
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
  (void)residual_norms;
  (void)residual_scales;
  (void)scale_resid;
  throw std::logic_error("score_queries_permute_dot is only available on x86_64");
#endif
}

}  // namespace vectorcache
