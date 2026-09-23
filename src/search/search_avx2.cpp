#include "vectorcache/search/search_avx2.hpp"

#include <algorithm>
#include <cstring>

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
                                 const std::uint64_t* id_map) {
  const __m256i nibble_mask = _mm256_set1_epi8(static_cast<char>(0x0F));
  const __m256 v_scale = _mm256_set1_ps(lut.scale);
  const float bias = lut.bias + bias_corr;
  const std::uint8_t* codes_base = blocked_codes.data();

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    const std::size_t end = std::min(base_vec + kBlock, n_vectors);

    __m256 fa[4] = {
        _mm256_set1_ps(bias),
        _mm256_set1_ps(bias),
        _mm256_set1_ps(bias),
        _mm256_set1_ps(bias),
    };

    const std::size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
      const std::size_t g_start = batch * kFlushEvery;
      const std::size_t g_end = std::min(g_start + kFlushEvery, n_byte_groups);
      __m256i accus[4] = {
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
          _mm256_setzero_si256(),
      };

      for (std::size_t g = g_start; g < g_end; ++g) {
        const std::uint8_t* cp = codes_base + (b * n_byte_groups + g) * kBlock;
        const __m256i codes_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp));
        const __m256i clo = _mm256_and_si256(codes_v, nibble_mask);
        const __m256i chi =
            _mm256_and_si256(_mm256_srli_epi16(codes_v, 4), nibble_mask);

        const __m256i lut_v =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(lut.uint8_luts + g * 32));
        const __m256i res0 = _mm256_shuffle_epi8(lut_v, clo);
        const __m256i res1 = _mm256_shuffle_epi8(lut_v, chi);
        accus[0] = _mm256_add_epi16(accus[0], res0);
        accus[1] = _mm256_add_epi16(accus[1], _mm256_srli_epi16(res0, 8));
        accus[2] = _mm256_add_epi16(accus[2], res1);
        accus[3] = _mm256_add_epi16(accus[3], _mm256_srli_epi16(res1, 8));
      }
      flush_batch_to_fa(accus, v_scale, fa);
    }

    alignas(32) float block_out[kBlock];
    if (end - base_vec == kBlock) {
      _mm256_storeu_ps(block_out,
                       _mm256_mul_ps(fa[0], _mm256_loadu_ps(vec_scales.data() + base_vec)));
      _mm256_storeu_ps(block_out + 8,
                       _mm256_mul_ps(fa[1], _mm256_loadu_ps(vec_scales.data() + base_vec + 8)));
      _mm256_storeu_ps(block_out + 16,
                       _mm256_mul_ps(fa[2], _mm256_loadu_ps(vec_scales.data() + base_vec + 16)));
      _mm256_storeu_ps(block_out + 24,
                       _mm256_mul_ps(fa[3], _mm256_loadu_ps(vec_scales.data() + base_vec + 24)));
    } else {
      _mm256_storeu_ps(block_out, fa[0]);
      _mm256_storeu_ps(block_out + 8, fa[1]);
      _mm256_storeu_ps(block_out + 16, fa[2]);
      _mm256_storeu_ps(block_out + 24, fa[3]);
      for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
        block_out[lane] *= vec_scales[base_vec + lane];
      }
    }

    for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
      heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k, block_out[lane],
                           topk_map_id(id_map, base_vec + lane));
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
                            const std::uint64_t* id_map) {
#if defined(__x86_64__) || defined(_M_X64)
  score_query_avx2_perm0_impl(lut, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, k,
                              heap_s, heap_i, heap_sz, heap_min, heap_mi, bias_corr, id_map);
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
#endif
}

}  // namespace vectorcache
