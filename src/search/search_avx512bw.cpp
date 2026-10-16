#include "vectorcache/search/search_avx512bw.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "vectorcache/constants.hpp"
#include "vectorcache/search/topk_heap.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace vectorcache {
namespace {

#if defined(__x86_64__) || defined(_M_X64)

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

void store_block_scores(const __m256 fa[4], std::size_t base_vec, std::size_t end,
                        const float* vec_scales, float* block_out) {
  if (end - base_vec == kBlock) {
    _mm256_storeu_ps(block_out, _mm256_mul_ps(fa[0], _mm256_loadu_ps(vec_scales + base_vec)));
    _mm256_storeu_ps(block_out + 8,
                     _mm256_mul_ps(fa[1], _mm256_loadu_ps(vec_scales + base_vec + 8)));
    _mm256_storeu_ps(block_out + 16,
                     _mm256_mul_ps(fa[2], _mm256_loadu_ps(vec_scales + base_vec + 16)));
    _mm256_storeu_ps(block_out + 24,
                     _mm256_mul_ps(fa[3], _mm256_loadu_ps(vec_scales + base_vec + 24)));
  } else {
    _mm256_storeu_ps(block_out, fa[0]);
    _mm256_storeu_ps(block_out + 8, fa[1]);
    _mm256_storeu_ps(block_out + 16, fa[2]);
    _mm256_storeu_ps(block_out + 24, fa[3]);
    for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
      block_out[lane] *= vec_scales[base_vec + lane];
    }
  }
}

void score_block_avx2_body(const QueryLutView* luts, std::size_t nq, const std::uint8_t* codes_base,
                           std::size_t b, std::size_t n_byte_groups, const __m256* v_scale,
                           const float* bias, __m256 fa[][4]) {
  const __m256i nibble_mask = _mm256_set1_epi8(static_cast<char>(0x0F));
  for (std::size_t qi = 0; qi < nq; ++qi) {
    fa[qi][0] = _mm256_set1_ps(bias[qi]);
    fa[qi][1] = _mm256_set1_ps(bias[qi]);
    fa[qi][2] = _mm256_set1_ps(bias[qi]);
    fa[qi][3] = _mm256_set1_ps(bias[qi]);
  }
  const std::size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
  for (std::size_t batch = 0; batch < n_batches; ++batch) {
    const std::size_t g_start = batch * kFlushEvery;
    const std::size_t g_end = std::min(g_start + kFlushEvery, n_byte_groups);
    __m256i accus[kAvx512bwNqBatch][4];
    for (std::size_t qi = 0; qi < nq; ++qi) {
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
      for (std::size_t qi = 0; qi < nq; ++qi) {
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
    for (std::size_t qi = 0; qi < nq; ++qi) {
      flush_batch_to_fa(accus[qi], v_scale[qi], fa[qi]);
    }
  }
}

void score_queries_avx512bw_perm0_impl(const QueryLutView* luts, std::size_t nq,
                                       std::span<const std::uint8_t> blocked_codes,
                                       std::span<const float> vec_scales, std::size_t n_byte_groups,
                                       std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                       float* const* heap_s, std::uint64_t* const* heap_i,
                                       std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                       const float* bias_corrs, const std::uint64_t* id_map,
                                       const float* score_offsets) {
  const __m512i mask512 = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::uint8_t* codes_base = blocked_codes.data();
  const std::size_t n_block_pairs = n_blocks / 2;

  __m256 v_scale[kAvx512bwNqBatch];
  float bias[kAvx512bwNqBatch];
  float score_off[kAvx512bwNqBatch];
  for (std::size_t qi = 0; qi < nq; ++qi) {
    v_scale[qi] = _mm256_set1_ps(luts[qi].scale);
    const float bc = bias_corrs ? bias_corrs[qi] : 0.f;
    bias[qi] = luts[qi].bias + bc;
    score_off[qi] = score_offsets ? score_offsets[qi] : 0.f;
  }

  auto push_block = [&](const __m256 fa[4], std::size_t base_vec, std::size_t qi) {
    const std::size_t end = std::min(base_vec + kBlock, n_vectors);
    if (base_vec >= n_vectors) return;
    alignas(32) float block_out[kBlock];
    store_block_scores(fa, base_vec, end, vec_scales.data(), block_out);
    for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
      heap_push_or_replace(heap_s[qi], heap_i[qi], heap_sz[qi], heap_min[qi], heap_mi[qi], k,
                           block_out[lane] + score_off[qi],
                           topk_map_id(id_map, base_vec + lane));
    }
  };

  for (std::size_t p = 0; p < n_block_pairs; ++p) {
    const std::size_t b0 = p * 2;
    const std::size_t b1 = b0 + 1;

    __m256 fa_b0[kAvx512bwNqBatch][4];
    __m256 fa_b1[kAvx512bwNqBatch][4];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      for (int i = 0; i < 4; ++i) {
        fa_b0[qi][i] = _mm256_set1_ps(bias[qi]);
        fa_b1[qi][i] = _mm256_set1_ps(bias[qi]);
      }
    }

    const std::size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
      const std::size_t g_start = batch * kFlushEvery;
      const std::size_t g_end = std::min(g_start + kFlushEvery, n_byte_groups);
      __m512i accus[kAvx512bwNqBatch][4];
      for (std::size_t qi = 0; qi < nq; ++qi) {
        for (int i = 0; i < 4; ++i) accus[qi][i] = _mm512_setzero_si512();
      }

      std::size_t g_pair = g_start;
      while (g_pair + 1 < g_end) {
        const std::size_t g0 = g_pair;
        const std::size_t g1 = g0 + 1;
        const std::uint8_t* cp0_a = codes_base + (b0 * n_byte_groups + g0) * kBlock;
        const std::uint8_t* cp1_a = codes_base + (b1 * n_byte_groups + g0) * kBlock;
        const __m512i codes_a = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0_a))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1_a)), 1);
        const std::uint8_t* cp0_b = codes_base + (b0 * n_byte_groups + g1) * kBlock;
        const std::uint8_t* cp1_b = codes_base + (b1 * n_byte_groups + g1) * kBlock;
        const __m512i codes_b = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0_b))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1_b)), 1);

        const __m512i clo_a = _mm512_and_si512(codes_a, mask512);
        const __m512i chi_a = _mm512_and_si512(_mm512_srli_epi16(codes_a, 4), mask512);
        const __m512i clo_b = _mm512_and_si512(codes_b, mask512);
        const __m512i chi_b = _mm512_and_si512(_mm512_srli_epi16(codes_b, 4), mask512);

        for (std::size_t qi = 0; qi < nq; ++qi) {
          const __m512i lut_a = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g0 * 32)));
          const __m512i lut_b = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g1 * 32)));
          const __m512i res0_a = _mm512_shuffle_epi8(lut_a, clo_a);
          const __m512i res1_a = _mm512_shuffle_epi8(lut_a, chi_a);
          const __m512i res0_b = _mm512_shuffle_epi8(lut_b, clo_b);
          const __m512i res1_b = _mm512_shuffle_epi8(lut_b, chi_b);
          accus[qi][0] = _mm512_add_epi16(accus[qi][0], _mm512_add_epi16(res0_a, res0_b));
          accus[qi][1] = _mm512_add_epi16(
              accus[qi][1],
              _mm512_add_epi16(_mm512_srli_epi16(res0_a, 8), _mm512_srli_epi16(res0_b, 8)));
          accus[qi][2] = _mm512_add_epi16(accus[qi][2], _mm512_add_epi16(res1_a, res1_b));
          accus[qi][3] = _mm512_add_epi16(
              accus[qi][3],
              _mm512_add_epi16(_mm512_srli_epi16(res1_a, 8), _mm512_srli_epi16(res1_b, 8)));
        }
        g_pair += 2;
      }

      for (std::size_t g = g_pair; g < g_end; ++g) {
        const std::uint8_t* cp0 = codes_base + (b0 * n_byte_groups + g) * kBlock;
        const std::uint8_t* cp1 = codes_base + (b1 * n_byte_groups + g) * kBlock;
        const __m512i codes_v = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1)), 1);
        const __m512i clo = _mm512_and_si512(codes_v, mask512);
        const __m512i chi = _mm512_and_si512(_mm512_srli_epi16(codes_v, 4), mask512);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          const __m512i lut = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g * 32)));
          const __m512i res0 = _mm512_shuffle_epi8(lut, clo);
          const __m512i res1 = _mm512_shuffle_epi8(lut, chi);
          accus[qi][0] = _mm512_add_epi16(accus[qi][0], res0);
          accus[qi][1] = _mm512_add_epi16(accus[qi][1], _mm512_srli_epi16(res0, 8));
          accus[qi][2] = _mm512_add_epi16(accus[qi][2], res1);
          accus[qi][3] = _mm512_add_epi16(accus[qi][3], _mm512_srli_epi16(res1, 8));
        }
      }

      for (std::size_t qi = 0; qi < nq; ++qi) {
        const __m256i block_b0[4] = {_mm512_castsi512_si256(accus[qi][0]),
                                     _mm512_castsi512_si256(accus[qi][1]),
                                     _mm512_castsi512_si256(accus[qi][2]),
                                     _mm512_castsi512_si256(accus[qi][3])};
        flush_batch_to_fa(block_b0, v_scale[qi], fa_b0[qi]);
        const __m256i block_b1[4] = {_mm512_extracti64x4_epi64(accus[qi][0], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][1], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][2], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][3], 1)};
        flush_batch_to_fa(block_b1, v_scale[qi], fa_b1[qi]);
      }
    }

    for (std::size_t qi = 0; qi < nq; ++qi) {
      push_block(fa_b0[qi], b0 * kBlock, qi);
      push_block(fa_b1[qi], b1 * kBlock, qi);
    }
  }

  const std::size_t bulk = n_block_pairs * 2;
  if (bulk < n_blocks) {
    __m256 fa[kAvx512bwNqBatch][4];
    score_block_avx2_body(luts, nq, codes_base, bulk, n_byte_groups, v_scale, bias, fa);
    for (std::size_t qi = 0; qi < nq; ++qi) {
      push_block(fa[qi], bulk * kBlock, qi);
    }
  }
}

void score_queries_avx512bw_perm0_threshold_fill_impl(
    const QueryLutView* luts, std::size_t nq, std::span<const std::uint8_t> blocked_codes,
    std::span<const float> vec_scales, std::size_t n_byte_groups, std::size_t n_vectors,
    std::size_t n_blocks, float threshold, std::size_t capacity, float* const* out_scores,
    std::uint64_t* const* out_ids, std::size_t* out_counts, const float* bias_corrs,
    const std::uint64_t* id_map, const float* score_offsets) {
  const bool count_only = out_scores == nullptr;
  const __m512i mask512 = _mm512_set1_epi8(static_cast<char>(0x0F));
  const std::uint8_t* codes_base = blocked_codes.data();
  const std::size_t n_block_pairs = n_blocks / 2;

  __m256 v_scale[kAvx512bwNqBatch];
  float bias[kAvx512bwNqBatch];
  float score_off[kAvx512bwNqBatch];
  for (std::size_t qi = 0; qi < nq; ++qi) {
    v_scale[qi] = _mm256_set1_ps(luts[qi].scale);
    const float bc = bias_corrs ? bias_corrs[qi] : 0.f;
    bias[qi] = luts[qi].bias + bc;
    score_off[qi] = score_offsets ? score_offsets[qi] : 0.f;
  }

  auto all_full = [&]() {
    if (count_only) return false;
    for (std::size_t qi = 0; qi < nq; ++qi) {
      if (out_counts[qi] < capacity) return false;
    }
    return true;
  };

  auto collect_block = [&](const __m256 fa[4], std::size_t base_vec, std::size_t qi) {
    if (!count_only && out_counts[qi] >= capacity) return;
    const std::size_t end = std::min(base_vec + kBlock, n_vectors);
    if (base_vec >= n_vectors) return;
    alignas(32) float block_out[kBlock];
    store_block_scores(fa, base_vec, end, vec_scales.data(), block_out);
    for (std::size_t lane = 0; lane < end - base_vec; ++lane) {
      if (!count_only && out_counts[qi] >= capacity) break;
      const float score = block_out[lane] + score_off[qi];
      if (score >= threshold) {
        if (!count_only) {
          const std::size_t c = out_counts[qi];
          out_scores[qi][c] = score;
          out_ids[qi][c] = topk_map_id(id_map, base_vec + lane);
        }
        ++out_counts[qi];
      }
    }
  };

  for (std::size_t p = 0; p < n_block_pairs; ++p) {
    if (all_full()) break;
    const std::size_t b0 = p * 2;
    const std::size_t b1 = b0 + 1;

    __m256 fa_b0[kAvx512bwNqBatch][4];
    __m256 fa_b1[kAvx512bwNqBatch][4];
    for (std::size_t qi = 0; qi < nq; ++qi) {
      for (int i = 0; i < 4; ++i) {
        fa_b0[qi][i] = _mm256_set1_ps(bias[qi]);
        fa_b1[qi][i] = _mm256_set1_ps(bias[qi]);
      }
    }

    const std::size_t n_batches = (n_byte_groups + kFlushEvery - 1) / kFlushEvery;
    for (std::size_t batch = 0; batch < n_batches; ++batch) {
      const std::size_t g_start = batch * kFlushEvery;
      const std::size_t g_end = std::min(g_start + kFlushEvery, n_byte_groups);
      __m512i accus[kAvx512bwNqBatch][4];
      for (std::size_t qi = 0; qi < nq; ++qi) {
        for (int i = 0; i < 4; ++i) accus[qi][i] = _mm512_setzero_si512();
      }

      std::size_t g_pair = g_start;
      while (g_pair + 1 < g_end) {
        const std::size_t g0 = g_pair;
        const std::size_t g1 = g0 + 1;
        const std::uint8_t* cp0_a = codes_base + (b0 * n_byte_groups + g0) * kBlock;
        const std::uint8_t* cp1_a = codes_base + (b1 * n_byte_groups + g0) * kBlock;
        const __m512i codes_a = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0_a))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1_a)), 1);
        const std::uint8_t* cp0_b = codes_base + (b0 * n_byte_groups + g1) * kBlock;
        const std::uint8_t* cp1_b = codes_base + (b1 * n_byte_groups + g1) * kBlock;
        const __m512i codes_b = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0_b))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1_b)), 1);
        const __m512i clo_a = _mm512_and_si512(codes_a, mask512);
        const __m512i chi_a = _mm512_and_si512(_mm512_srli_epi16(codes_a, 4), mask512);
        const __m512i clo_b = _mm512_and_si512(codes_b, mask512);
        const __m512i chi_b = _mm512_and_si512(_mm512_srli_epi16(codes_b, 4), mask512);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          if (!count_only && out_counts[qi] >= capacity) continue;
          const __m512i lut_a = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g0 * 32)));
          const __m512i lut_b = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g1 * 32)));
          const __m512i res0_a = _mm512_shuffle_epi8(lut_a, clo_a);
          const __m512i res1_a = _mm512_shuffle_epi8(lut_a, chi_a);
          const __m512i res0_b = _mm512_shuffle_epi8(lut_b, clo_b);
          const __m512i res1_b = _mm512_shuffle_epi8(lut_b, chi_b);
          accus[qi][0] = _mm512_add_epi16(accus[qi][0], _mm512_add_epi16(res0_a, res0_b));
          accus[qi][1] = _mm512_add_epi16(
              accus[qi][1],
              _mm512_add_epi16(_mm512_srli_epi16(res0_a, 8), _mm512_srli_epi16(res0_b, 8)));
          accus[qi][2] = _mm512_add_epi16(accus[qi][2], _mm512_add_epi16(res1_a, res1_b));
          accus[qi][3] = _mm512_add_epi16(
              accus[qi][3],
              _mm512_add_epi16(_mm512_srli_epi16(res1_a, 8), _mm512_srli_epi16(res1_b, 8)));
        }
        g_pair += 2;
      }
      for (std::size_t g = g_pair; g < g_end; ++g) {
        const std::uint8_t* cp0 = codes_base + (b0 * n_byte_groups + g) * kBlock;
        const std::uint8_t* cp1 = codes_base + (b1 * n_byte_groups + g) * kBlock;
        const __m512i codes_v = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp0))),
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(cp1)), 1);
        const __m512i clo = _mm512_and_si512(codes_v, mask512);
        const __m512i chi = _mm512_and_si512(_mm512_srli_epi16(codes_v, 4), mask512);
        for (std::size_t qi = 0; qi < nq; ++qi) {
          if (!count_only && out_counts[qi] >= capacity) continue;
          const __m512i lut = _mm512_broadcast_i64x4(
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(luts[qi].uint8_luts + g * 32)));
          const __m512i res0 = _mm512_shuffle_epi8(lut, clo);
          const __m512i res1 = _mm512_shuffle_epi8(lut, chi);
          accus[qi][0] = _mm512_add_epi16(accus[qi][0], res0);
          accus[qi][1] = _mm512_add_epi16(accus[qi][1], _mm512_srli_epi16(res0, 8));
          accus[qi][2] = _mm512_add_epi16(accus[qi][2], res1);
          accus[qi][3] = _mm512_add_epi16(accus[qi][3], _mm512_srli_epi16(res1, 8));
        }
      }
      for (std::size_t qi = 0; qi < nq; ++qi) {
        if (!count_only && out_counts[qi] >= capacity) continue;
        const __m256i block_b0[4] = {_mm512_castsi512_si256(accus[qi][0]),
                                     _mm512_castsi512_si256(accus[qi][1]),
                                     _mm512_castsi512_si256(accus[qi][2]),
                                     _mm512_castsi512_si256(accus[qi][3])};
        flush_batch_to_fa(block_b0, v_scale[qi], fa_b0[qi]);
        const __m256i block_b1[4] = {_mm512_extracti64x4_epi64(accus[qi][0], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][1], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][2], 1),
                                     _mm512_extracti64x4_epi64(accus[qi][3], 1)};
        flush_batch_to_fa(block_b1, v_scale[qi], fa_b1[qi]);
      }
    }

    for (std::size_t qi = 0; qi < nq; ++qi) {
      collect_block(fa_b0[qi], b0 * kBlock, qi);
      collect_block(fa_b1[qi], b1 * kBlock, qi);
    }
  }

  const std::size_t bulk = n_block_pairs * 2;
  if (bulk < n_blocks && !all_full()) {
    __m256 fa[kAvx512bwNqBatch][4];
    score_block_avx2_body(luts, nq, codes_base, bulk, n_byte_groups, v_scale, bias, fa);
    for (std::size_t qi = 0; qi < nq; ++qi) {
      collect_block(fa[qi], bulk * kBlock, qi);
    }
  }
}

#endif  // x86_64

}  // namespace

void score_queries_avx512bw_perm0(const QueryLutView* luts, std::size_t nq,
                                  std::span<const std::uint8_t> blocked_codes,
                                  std::span<const float> vec_scales, std::size_t n_byte_groups,
                                  std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                  float* const* heap_s, std::uint64_t* const* heap_i,
                                  std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                  const float* bias_corrs, const std::uint64_t* id_map,
                                  const float* score_offsets) {
  if (nq == 0) return;
  if (nq > kAvx512bwNqBatch) {
    throw std::invalid_argument("score_queries_avx512bw_perm0: nq > kAvx512bwNqBatch");
  }
#if defined(__x86_64__) || defined(_M_X64)
  score_queries_avx512bw_perm0_impl(luts, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors,
                                    n_blocks, k, heap_s, heap_i, heap_sz, heap_min, heap_mi,
                                    bias_corrs, id_map, score_offsets);
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
  throw std::logic_error("score_queries_avx512bw_perm0 requires x86_64");
#endif
}

void score_queries_avx512bw_perm0_threshold_fill(
    const QueryLutView* luts, std::size_t nq, std::span<const std::uint8_t> blocked_codes,
    std::span<const float> vec_scales, std::size_t n_byte_groups, std::size_t n_vectors,
    std::size_t n_blocks, float threshold, std::size_t capacity, float* const* out_scores,
    std::uint64_t* const* out_ids, std::size_t* out_counts, const float* bias_corrs,
    const std::uint64_t* id_map, const float* score_offsets) {
  if (nq == 0) return;
  if (nq > kAvx512bwNqBatch) {
    throw std::invalid_argument("score_queries_avx512bw_perm0_threshold_fill: nq > batch");
  }
#if defined(__x86_64__) || defined(_M_X64)
  score_queries_avx512bw_perm0_threshold_fill_impl(
      luts, nq, blocked_codes, vec_scales, n_byte_groups, n_vectors, n_blocks, threshold, capacity,
      out_scores, out_ids, out_counts, bias_corrs, id_map, score_offsets);
#else
  (void)luts;
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
  (void)bias_corrs;
  (void)id_map;
  (void)score_offsets;
  throw std::logic_error("score_queries_avx512bw_perm0_threshold_fill requires x86_64");
#endif
}

}  // namespace vectorcache
