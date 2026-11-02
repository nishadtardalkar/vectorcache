#include "vectorcache/search/float_rescore.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

#if defined(__x86_64__) || defined(_M_X64)
inline float hsum256_ps(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 sh = _mm_movehdup_ps(lo);
  lo = _mm_add_ps(lo, sh);
  sh = _mm_movehl_ps(sh, lo);
  lo = _mm_add_ss(lo, sh);
  return _mm_cvtss_f32(lo);
}

/// Unit·unit IP; `dim` is a multiple of 8 in this codebase.
inline float dot_f32_avx2(const float* a, const float* b, std::size_t dim) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  std::size_t i = 0;
  // Two-accumulator FMA to hide latency; dim=1536 → 96 iters of 16 floats.
  for (; i + 16 <= dim; i += 16) {
    const __m256 a0 = _mm256_loadu_ps(a + i);
    const __m256 b0 = _mm256_loadu_ps(b + i);
    const __m256 a1 = _mm256_loadu_ps(a + i + 8);
    const __m256 b1 = _mm256_loadu_ps(b + i + 8);
    acc0 = _mm256_fmadd_ps(a0, b0, acc0);
    acc1 = _mm256_fmadd_ps(a1, b1, acc1);
  }
  acc0 = _mm256_add_ps(acc0, acc1);
  for (; i + 8 <= dim; i += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
  }
  float s = hsum256_ps(acc0);
  for (; i < dim; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

inline void prefetch_row(const float* row) {
  // Touch first few cache lines; row is up to ~6KB (dim=1536).
  _mm_prefetch(reinterpret_cast<const char*>(row), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(row) + 64, _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(row) + 128, _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(row) + 192, _MM_HINT_T0);
}
#else
inline float dot_f32_avx2(const float* a, const float* b, std::size_t dim) {
  double s = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(a[i]) * b[i];
  }
  return static_cast<float>(s);
}
inline void prefetch_row(const float*) {}
#endif

}  // namespace

void float_rescore_topk(SearchResults& out, std::span<const float> queries,
                        std::span<const float> corpus, std::size_t dim) {
  if (out.nq == 0 || out.k == 0 || dim == 0) {
    return;
  }
  if (queries.size() != out.nq * dim) {
    throw std::invalid_argument("float_rescore_topk: queries size mismatch");
  }
  if (corpus.size() % dim != 0) {
    throw std::invalid_argument("float_rescore_topk: corpus size must be multiple of dim");
  }
  const std::size_t n_corpus = corpus.size() / dim;
  const std::size_t nq = out.nq;
  const std::size_t k = out.k;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    const std::size_t qis = static_cast<std::size_t>(qi);
    const float* q = queries.data() + qis * dim;
    float* scores = out.scores.data() + qis * k;
    std::uint64_t* ids = out.ids.data() + qis * k;
    const std::size_t n =
        out.counts.empty() ? k : std::min(k, out.counts[qis]);
    if (n == 0) {
      continue;
    }

    // Prefetch first candidate; pipeline prefetch of i+1 while scoring i.
    {
      const std::size_t id0 = static_cast<std::size_t>(ids[0]);
      if (id0 < n_corpus) {
        prefetch_row(corpus.data() + id0 * dim);
      }
    }
    for (std::size_t j = 0; j < n; ++j) {
      if (j + 1 < n) {
        const std::size_t nid = static_cast<std::size_t>(ids[j + 1]);
        if (nid < n_corpus) {
          prefetch_row(corpus.data() + nid * dim);
        }
      }
      const std::size_t id = static_cast<std::size_t>(ids[j]);
      if (id >= n_corpus) {
        scores[j] = -std::numeric_limits<float>::infinity();
        continue;
      }
      scores[j] = dot_f32_avx2(q, corpus.data() + id * dim, dim);
    }

    // Small-k reorder without heap traffic (k≤256 covers published cells).
    constexpr std::size_t kStack = 256;
    std::size_t order_stack[kStack];
    float tmp_s_stack[kStack];
    std::uint64_t tmp_i_stack[kStack];
    std::vector<std::size_t> order_heap;
    std::vector<float> tmp_s_heap;
    std::vector<std::uint64_t> tmp_i_heap;
    std::size_t* order = order_stack;
    float* tmp_s = tmp_s_stack;
    std::uint64_t* tmp_i = tmp_i_stack;
    if (n > kStack) {
      order_heap.resize(n);
      tmp_s_heap.resize(n);
      tmp_i_heap.resize(n);
      order = order_heap.data();
      tmp_s = tmp_s_heap.data();
      tmp_i = tmp_i_heap.data();
    }
    for (std::size_t j = 0; j < n; ++j) {
      order[j] = j;
    }
    std::sort(order, order + n, [&](std::size_t a, std::size_t b) {
      if (scores[a] != scores[b]) return scores[a] > scores[b];
      return ids[a] < ids[b];
    });
    for (std::size_t j = 0; j < n; ++j) {
      tmp_s[j] = scores[order[j]];
      tmp_i[j] = ids[order[j]];
    }
    std::copy(tmp_s, tmp_s + n, scores);
    std::copy(tmp_i, tmp_i + n, ids);
  }
}

}  // namespace vectorcache
