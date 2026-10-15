#include "vectorcache/search/landmark_scan.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "vectorcache/search/topk_heap.hpp"

#if defined(__AVX2__) || defined(_MSC_VER)
#include <immintrin.h>
#define VECTORCACHE_LANDMARK_AVX2 1
#endif

namespace vectorcache {
namespace {

float dot_scalar(const float* a, const float* b, std::size_t n) {
  float s = 0.f;
  for (std::size_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

#if VECTORCACHE_LANDMARK_AVX2
float dot_avx(const float* a, const float* b, std::size_t n) {
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 va = _mm256_loadu_ps(a + i);
    const __m256 vb = _mm256_loadu_ps(b + i);
    acc = _mm256_fmadd_ps(va, vb, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}
#endif

float dot_f(const float* a, const float* b, std::size_t n) {
#if VECTORCACHE_LANDMARK_AVX2
  return dot_avx(a, b, n);
#else
  return dot_scalar(a, b, n);
#endif
}

}  // namespace

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

void exact_rerank_topk(std::span<const float> query_unit, std::size_t dim,
                       std::span<const float> db_unit, std::span<const std::uint64_t> cand_ids,
                       std::size_t n_cand, std::size_t k, float* out_scores,
                       std::uint64_t* out_ids) {
  const std::size_t kk = std::min(k, n_cand);
  std::vector<float> heap_s(kk);
  std::vector<std::uint64_t> heap_i(kk);
  std::size_t heap_sz = 0;
  float heap_min = 0.f;
  std::size_t heap_mi = 0;

  for (std::size_t c = 0; c < n_cand; ++c) {
    const std::uint64_t id = cand_ids[c];
    const float* row = db_unit.data() + static_cast<std::size_t>(id) * dim;
    const float score = dot_f(query_unit.data(), row, dim);
    heap_push_or_replace(heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi, kk, score, id);
  }

  // Extract heap to descending arrays via partial sort of pairs.
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
}

}  // namespace vectorcache
