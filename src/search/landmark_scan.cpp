#include "vectorcache/search/landmark_scan.hpp"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cstring>

#include "vectorcache/search/topk_heap.hpp"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define VECTORCACHE_LANDMARK_AVX2 1
#endif

namespace vectorcache {

std::vector<float> pack_landmark_blocks(std::span<const float> rows, std::size_t n, std::size_t m) {
  const std::size_t n_blocks = (n + kBlock - 1) / kBlock;
  std::vector<float> blocked(n_blocks * m * kBlock, 0.f);
  for (std::size_t b = 0; b < n_blocks; ++b) {
    float* blk = blocked.data() + b * m * kBlock;
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::size_t vi = b * kBlock + lane;
      if (vi >= n) {
        break;
      }
      const float* row = rows.data() + vi * m;
      for (std::size_t d = 0; d < m; ++d) {
        blk[d * kBlock + lane] = row[d];
      }
    }
  }
  return blocked;
}

void landmark_shortlist_scan(std::span<const float> q_landmarks, std::size_t m,
                             std::span<const float> blocked, std::size_t n_vectors,
                             std::size_t n_blocks, std::size_t shortlist, float* heap_s,
                             std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min) {
  heap_sz = 0;
  heap_min = 0.f;
  std::size_t heap_mi = 0;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const float* blk = blocked.data() + b * m * kBlock;
    alignas(32) float scores[kBlock];
#if VECTORCACHE_LANDMARK_AVX2
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    for (std::size_t d = 0; d < m; ++d) {
      const __m256 qv = _mm256_set1_ps(q_landmarks[d]);
      const float* col = blk + d * kBlock;
      acc0 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(col + 0), acc0);
      acc1 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(col + 8), acc1);
      acc2 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(col + 16), acc2);
      acc3 = _mm256_fmadd_ps(qv, _mm256_loadu_ps(col + 24), acc3);
    }
    _mm256_storeu_ps(scores + 0, acc0);
    _mm256_storeu_ps(scores + 8, acc1);
    _mm256_storeu_ps(scores + 16, acc2);
    _mm256_storeu_ps(scores + 24, acc3);
#else
    std::memset(scores, 0, sizeof(scores));
    for (std::size_t d = 0; d < m; ++d) {
      const float qv = q_landmarks[d];
      const float* col = blk + d * kBlock;
      for (std::size_t lane = 0; lane < kBlock; ++lane) {
        scores[lane] += qv * col[lane];
      }
    }
#endif
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::size_t vi = b * kBlock + lane;
      if (vi >= n_vectors) {
        break;
      }
      heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, shortlist, scores[lane],
                           static_cast<std::uint64_t>(vi));
    }
  }
}

void quantized_rerank_from_bitmap(std::span<const std::uint8_t> lut_bytes, float lut_scale,
                                  float lut_bias, std::span<const std::uint8_t> codes_flat,
                                  std::size_t n_byte_groups, std::span<const float> scales,
                                  const std::uint64_t* hit_bitmap, std::size_t n_vectors,
                                  std::size_t k, float* out_scores, std::uint64_t* out_ids,
                                  double* score_ms_out, double* select_ms_out) {
  using clock = std::chrono::steady_clock;
  const auto t_score0 = clock::now();

  const std::size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
  const std::size_t kk = std::min(k, n_vectors);
  std::vector<float> heap_s(kk);
  std::vector<std::uint64_t> heap_i(kk);
  std::size_t heap_sz = 0;
  float heap_min = 0.f;
  std::size_t heap_mi = 0;

  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base = b * kBlock;
    const std::size_t end = std::min(base + kBlock, n_vectors);
    // 32-bit window into the bitmap (kBlock == 32).
    const std::size_t word = base >> 6;
    const std::size_t bit = base & 63u;
    std::uint32_t mask;
    if (bit <= 32) {
      mask = static_cast<std::uint32_t>((hit_bitmap[word] >> bit) & 0xFFFFFFFFu);
    } else {
      // Spans two u64 words (shouldn't happen for base multiples of 32).
      mask = static_cast<std::uint32_t>(hit_bitmap[word] >> bit);
      mask |= static_cast<std::uint32_t>(hit_bitmap[word + 1] << (64u - bit));
    }
    // Clear bits past end of corpus.
    if (end - base < kBlock) {
      mask &= (std::uint32_t{1} << (end - base)) - 1u;
    }
    if (mask == 0) {
      continue;
    }
    while (mask != 0) {
      const unsigned lane = static_cast<unsigned>(std::countr_zero(mask));
      mask &= mask - 1u;
      const std::size_t vi = base + lane;
      const std::uint8_t* row = codes_flat.data() + vi * n_byte_groups;
      float score = lut_bias;
      for (std::size_t g = 0; g < n_byte_groups; ++g) {
        const auto byte_val = static_cast<std::size_t>(row[g]);
        score += lut_scale * static_cast<float>(lut_bytes[g * 32 + (byte_val >> 4)]);
        score += lut_scale * static_cast<float>(lut_bytes[g * 32 + 16 + (byte_val & 0x0F)]);
      }
      score *= scales[vi];
      heap_push_or_replace(heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi, kk, score,
                           static_cast<std::uint64_t>(vi));
    }
  }
  const double score_ms =
      std::chrono::duration<double, std::milli>(clock::now() - t_score0).count();

  const auto t_sel0 = clock::now();
  std::vector<std::pair<float, std::uint64_t>> pairs(heap_sz);
  for (std::size_t i = 0; i < heap_sz; ++i) {
    pairs[i] = {heap_s[i], heap_i[i]};
  }
  std::sort(pairs.begin(), pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });
  for (std::size_t i = 0; i < heap_sz; ++i) {
    out_scores[i] = pairs[i].first;
    out_ids[i] = pairs[i].second;
  }
  for (std::size_t i = heap_sz; i < k; ++i) {
    out_scores[i] = 0.f;
    out_ids[i] = 0;
  }
  const double select_ms =
      std::chrono::duration<double, std::milli>(clock::now() - t_sel0).count();
  if (score_ms_out) *score_ms_out += score_ms;
  if (select_ms_out) *select_ms_out += select_ms;
}

}  // namespace vectorcache
