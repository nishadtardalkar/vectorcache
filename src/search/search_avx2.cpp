#include "vectorcache/search/search_avx2.hpp"

#include <algorithm>
#include <cassert>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "vectorcache/constants.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/topk_heap.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vectorcache {
namespace {

#if defined(__x86_64__) || defined(_M_X64)

/// Flush one u16 accumulator batch into f32 scores (turbovec avx2_batch_flush_to_fa).
void flush_batch_to_fa(const __m256i accus[4], __m256 v_scale, __m256 fa[4]) {
  const __m256i a0 = _mm256_sub_epi16(accus[0], _mm256_slli_epi16(accus[1], 8));
  const __m256i a1 = accus[1];
  const __m256i a2 = _mm256_sub_epi16(accus[2], _mm256_slli_epi16(accus[3], 8));
  const __m256i a3 = accus[3];

  const __m256i dis0 = _mm256_add_epi16(_mm256_permute2x128_si256(a0, a1, 0x21),
                                        _mm256_blend_epi32(a0, a1, 0xF0));
  const __m256i dis1 = _mm256_add_epi16(_mm256_permute2x128_si256(a2, a3, 0x21),
                                        _mm256_blend_epi32(a2, a3, 0xF0));

  const __m256 f0 =
      _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis0)));
  const __m256 f1 =
      _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis0, 1)));
  const __m256 f2 =
      _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(dis1)));
  const __m256 f3 =
      _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(dis1, 1)));

  fa[0] = _mm256_fmadd_ps(v_scale, f0, fa[0]);
  fa[1] = _mm256_fmadd_ps(v_scale, f1, fa[1]);
  fa[2] = _mm256_fmadd_ps(v_scale, f2, fa[2]);
  fa[3] = _mm256_fmadd_ps(v_scale, f3, fa[3]);
}

void score_query_avx2_perm0_impl(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                                 std::span<const float> vec_scales, std::size_t n_byte_groups,
                                 std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                 float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                 float& heap_min, std::size_t& heap_mi, float bias_corr,
                                 const std::uint64_t* id_map, float score_offset) {
  QueryLutView luts[1] = {lut};
  float* hs[1] = {heap_s};
  std::uint64_t* hi[1] = {heap_i};
  std::size_t hsz[1] = {heap_sz};
  float hmin[1] = {heap_min};
  std::size_t hmi[1] = {heap_mi};
  const float bc[1] = {bias_corr};
  const float off[1] = {score_offset};
  score_queries_avx2_perm0(luts, 1, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                           hs, hi, hsz, hmin, hmi, bc, id_map, off);
  heap_sz = hsz[0];
  heap_min = hmin[0];
  heap_mi = hmi[0];
}

void score_queries_avx2_perm0_impl(const QueryLutView* luts, std::size_t nq,
                                   std::span<const std::uint8_t> blocked_codes,
                                   std::span<const float> vec_scales, std::size_t n_byte_groups,
                                   std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                   float* const* heap_s, std::uint64_t* const* heap_i,
                                   std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                   const float* bias_corrs, const std::uint64_t* id_map,
                                   const float* score_offsets) {
  const __m256i nibble_mask = _mm256_set1_epi8(static_cast<char>(0x0F));
  const std::uint8_t* codes_base = blocked_codes.data();

  // Caller pads `luts` to kAvx2NqBatch (turbovec always scores 4 lanes).
  assert(nq >= 1 && nq <= kAvx2NqBatch);
  __m256 v_scale[kAvx2NqBatch];
  float bias[kAvx2NqBatch];
  float score_off[kAvx2NqBatch];
  for (std::size_t qi = 0; qi < kAvx2NqBatch; ++qi) {
    v_scale[qi] = _mm256_set1_ps(luts[qi].scale);
    const float bc = bias_corrs ? bias_corrs[qi] : 0.f;
    bias[qi] = luts[qi].bias + bc;
    score_off[qi] = score_offsets ? score_offsets[qi] : 0.f;
  }

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t end = std::min(base_vec + kBlock, n_vectors);

    __m256 fa[kAvx2NqBatch][4];
    for (std::size_t qi = 0; qi < kAvx2NqBatch; ++qi) {
      fa[qi][0] = _mm256_set1_ps(bias[qi]);
      fa[qi][1] = _mm256_set1_ps(bias[qi]);
      fa[qi][2] = _mm256_set1_ps(bias[qi]);
      fa[qi][3] = _mm256_set1_ps(bias[qi]);
    }

    const std::size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
      const std::size_t g_start = batch * kFlushEvery;
      const std::size_t g_end = std::min(g_start + kFlushEvery, n_byte_groups);
      __m256i accus[kAvx2NqBatch][4];
      for (std::size_t qi = 0; qi < kAvx2NqBatch; ++qi) {
        accus[qi][0] = _mm256_setzero_si256();
        accus[qi][1] = _mm256_setzero_si256();
        accus[qi][2] = _mm256_setzero_si256();
        accus[qi][3] = _mm256_setzero_si256();
      }

      for (std::size_t g = g_start; g < g_end; ++g) {
        const std::uint8_t* cp = codes_base + (b * n_byte_groups + g) * kBlock;
        const __m256i codes_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp));
        const __m256i clo = _mm256_and_si256(codes_v, nibble_mask);
        const __m256i chi = _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), nibble_mask);

        for (std::size_t qi = 0; qi < kAvx2NqBatch; ++qi) {
          const __m256i lut_v =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g * 32));
          const __m256i res0 = _mm256_shuffle_epi8(lut_v, clo);
          const __m256i res1 = _mm256_shuffle_epi8(lut_v, chi);
          accus[qi][0] = _mm256_add_epi16(accus[qi][0], res0);
          accus[qi][1] = _mm256_add_epi16(accus[qi][1], _mm256_srli_epi16(res0, 8));
          accus[qi][2] = _mm256_add_epi16(accus[qi][2], res1);
          accus[qi][3] = _mm256_add_epi16(accus[qi][3], _mm256_srli_epi16(res1, 8));
        }
      }
      for (std::size_t qi = 0; qi < kAvx2NqBatch; ++qi) {
        flush_batch_to_fa(accus[qi], v_scale[qi], fa[qi]);
      }
    }

    for (std::size_t qi = 0; qi < nq; ++qi) {
      const std::size_t end_lane = end - base_vec;
      const float* scales_ptr = vec_scales.data() + base_vec;
      // turbovec avx2_post_flush_heap_update: early-out when heap full and no lane beats min.
      if (heap_sz[qi] >= k && end_lane == kBlock) {
        __m256 s0 = _mm256_mul_ps(fa[qi][0], _mm256_loadu_ps(scales_ptr));
        __m256 s1 = _mm256_mul_ps(fa[qi][1], _mm256_loadu_ps(scales_ptr + 8));
        __m256 s2 = _mm256_mul_ps(fa[qi][2], _mm256_loadu_ps(scales_ptr + 16));
        __m256 s3 = _mm256_mul_ps(fa[qi][3], _mm256_loadu_ps(scales_ptr + 24));
        if (score_off[qi] != 0.f) {
          const __m256 off = _mm256_set1_ps(score_off[qi]);
          s0 = _mm256_add_ps(s0, off);
          s1 = _mm256_add_ps(s1, off);
          s2 = _mm256_add_ps(s2, off);
          s3 = _mm256_add_ps(s3, off);
        }
        const __m256 thr = _mm256_set1_ps(heap_min[qi]);
        const std::uint32_t m0 =
            static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(s0, thr, _CMP_GT_OQ)));
        const std::uint32_t m1 =
            static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(s1, thr, _CMP_GT_OQ)));
        const std::uint32_t m2 =
            static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(s2, thr, _CMP_GT_OQ)));
        const std::uint32_t m3 =
            static_cast<std::uint32_t>(_mm256_movemask_ps(_mm256_cmp_ps(s3, thr, _CMP_GT_OQ)));
        if ((m0 | m1 | m2 | m3) == 0) {
          continue;
        }
        alignas(32) float block_out[kBlock];
        if (m0 != 0) _mm256_storeu_ps(block_out, s0);
        if (m1 != 0) _mm256_storeu_ps(block_out + 8, s1);
        if (m2 != 0) _mm256_storeu_ps(block_out + 16, s2);
        if (m3 != 0) _mm256_storeu_ps(block_out + 24, s3);
        const std::uint32_t masks[4] = {m0, m1, m2, m3};
        for (std::size_t chunk = 0; chunk < 4; ++chunk) {
          std::uint32_t m = masks[chunk];
          while (m != 0) {
            const unsigned bit = static_cast<unsigned>(std::countr_zero(m));
            m &= m - 1;
            const std::size_t lane = chunk * 8 + bit;
            const float score = block_out[lane];
            if (score > heap_min[qi]) {
              heap_push_or_replace(heap_s[qi], heap_i[qi], heap_sz[qi], heap_min[qi], heap_mi[qi],
                                   k, score, topk_map_id(id_map, base_vec + lane));
            }
          }
        }
        continue;
      }

      alignas(32) float block_out[kBlock];
      if (end_lane == kBlock) {
        _mm256_storeu_ps(block_out, _mm256_mul_ps(fa[qi][0], _mm256_loadu_ps(scales_ptr)));
        _mm256_storeu_ps(block_out + 8, _mm256_mul_ps(fa[qi][1], _mm256_loadu_ps(scales_ptr + 8)));
        _mm256_storeu_ps(block_out + 16,
                         _mm256_mul_ps(fa[qi][2], _mm256_loadu_ps(scales_ptr + 16)));
        _mm256_storeu_ps(block_out + 24,
                         _mm256_mul_ps(fa[qi][3], _mm256_loadu_ps(scales_ptr + 24)));
      } else {
        _mm256_storeu_ps(block_out, fa[qi][0]);
        _mm256_storeu_ps(block_out + 8, fa[qi][1]);
        _mm256_storeu_ps(block_out + 16, fa[qi][2]);
        _mm256_storeu_ps(block_out + 24, fa[qi][3]);
        for (std::size_t lane = 0; lane < end_lane; ++lane) {
          block_out[lane] *= scales_ptr[lane];
        }
      }

      for (std::size_t lane = 0; lane < end_lane; ++lane) {
        heap_push_or_replace(heap_s[qi], heap_i[qi], heap_sz[qi], heap_min[qi], heap_mi[qi], k,
                             block_out[lane] + score_off[qi],
                             topk_map_id(id_map, base_vec + lane));
      }
    }
  }
}

#endif  // x86_64

}  // namespace

void score_query_avx2_perm0(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                            std::span<const float> vec_scales, std::size_t n_byte_groups,
                            std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                            float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                            float& heap_min, std::size_t& heap_mi, float bias_corr,
                            const std::uint64_t* id_map, float score_offset) {
#if defined(__x86_64__) || defined(_M_X64)
  score_query_avx2_perm0_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
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
#endif
}

void score_queries_avx2_perm0(const QueryLutView* luts, std::size_t nq,
                              std::span<const std::uint8_t> blocked_codes,
                              std::span<const float> vec_scales, std::size_t n_byte_groups,
                              std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                              float* const* heap_s, std::uint64_t* const* heap_i,
                              std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                              const float* bias_corrs, const std::uint64_t* id_map,
                              const float* score_offsets) {
  if (nq == 0) {
    return;
  }
  if (nq > kAvx2NqBatch) {
    throw std::invalid_argument("score_queries_avx2_perm0: nq > kAvx2NqBatch");
  }
#if defined(__x86_64__) || defined(_M_X64)
  // Pad to kAvx2NqBatch so the accumulate loop is compile-time 4 (turbovec).
  QueryLutView pad_luts[kAvx2NqBatch];
  float* pad_heap_s[kAvx2NqBatch];
  std::uint64_t* pad_heap_i[kAvx2NqBatch];
  std::size_t pad_heap_sz[kAvx2NqBatch];
  float pad_heap_min[kAvx2NqBatch];
  std::size_t pad_heap_mi[kAvx2NqBatch];
  float pad_bias[kAvx2NqBatch];
  float pad_offs[kAvx2NqBatch];
  float dummy_s = 0.f;
  std::uint64_t dummy_i = 0;
  const std::size_t pad_qi = nq - 1;
  for (std::size_t i = 0; i < kAvx2NqBatch; ++i) {
    const std::size_t src = (i < nq) ? i : pad_qi;
    pad_luts[i] = luts[src];
    if (i < nq) {
      pad_heap_s[i] = heap_s[i];
      pad_heap_i[i] = heap_i[i];
      pad_heap_sz[i] = heap_sz[i];
      pad_heap_min[i] = heap_min[i];
      pad_heap_mi[i] = heap_mi[i];
    } else {
      pad_heap_s[i] = &dummy_s;
      pad_heap_i[i] = &dummy_i;
      pad_heap_sz[i] = 0;
      pad_heap_min[i] = -std::numeric_limits<float>::infinity();
      pad_heap_mi[i] = 0;
    }
    pad_bias[i] = bias_corrs ? bias_corrs[src] : 0.f;
    pad_offs[i] = score_offsets ? score_offsets[src] : 0.f;
  }
  score_queries_avx2_perm0_impl(pad_luts, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                n_blocks, k, pad_heap_s, pad_heap_i, pad_heap_sz, pad_heap_min,
                                pad_heap_mi, bias_corrs ? pad_bias : nullptr, id_map,
                                score_offsets ? pad_offs : nullptr);
  for (std::size_t i = 0; i < nq; ++i) {
    heap_sz[i] = pad_heap_sz[i];
    heap_min[i] = pad_heap_min[i];
    heap_mi[i] = pad_heap_mi[i];
  }
#else
  (void)luts;
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
  (void)bias_corrs;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_avx2_perm0 requires x86_64");
#endif
}

}  // namespace vectorcache
