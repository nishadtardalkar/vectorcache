#include "vectorcache/search/search.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/search/landmark_scan.hpp"
#include "vectorcache/search/topk_heap.hpp"
#include "vectorcache/transform/rotation.hpp"

#if defined(VECTORCACHE_OPENMP)
#include <omp.h>
#endif

namespace vectorcache {
namespace {

void l2_normalize_into(const float* src, float* dst, std::size_t dim) {
  double s = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(src[i]) * src[i];
  }
  const float nrm = static_cast<float>(std::sqrt(s));
  if (nrm < kMinInputNorm) {
    std::fill(dst, dst + dim, 0.f);
    return;
  }
  const float inv = 1.f / nrm;
  for (std::size_t i = 0; i < dim; ++i) {
    dst[i] = src[i] * inv;
  }
}

void project_landmarks(const float* rotated, std::size_t dim, std::span<const float> L,
                       std::size_t m, float* out) {
  for (std::size_t j = 0; j < m; ++j) {
    double s = 0.0;
    const float* Lj = L.data() + j * dim;
    for (std::size_t i = 0; i < dim; ++i) {
      s += static_cast<double>(rotated[i]) * Lj[i];
    }
    out[j] = static_cast<float>(s);
  }
}

}  // namespace

SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, std::size_t shortlist,
                                     std::span<const float> landmark_basis, std::size_t n_landmarks,
                                     std::span<const float> blocked_landmarks,
                                     std::size_t n_vectors, std::size_t n_blocks,
                                     std::span<const float> db_unit, const Rotation& rotation) {
  SearchResults out;
  out.nq = nq;
  out.k = k;
  out.scores.assign(nq * k, 0.f);
  out.ids.assign(nq * k, 0);

  if (nq == 0 || n_vectors == 0 || k == 0) {
    return out;
  }

  const std::size_t sl = std::min(shortlist, n_vectors);

#if defined(VECTORCACHE_OPENMP)
#pragma omp parallel
#endif
  {
    std::vector<float> q_unit(dim);
    std::vector<float> q_rot(dim);
    std::vector<float> q_land(n_landmarks);
    std::vector<float> heap_s(sl);
    std::vector<std::uint64_t> heap_i(sl);
    std::vector<std::uint64_t> cand(sl);

#if defined(VECTORCACHE_OPENMP)
#pragma omp for schedule(dynamic, 1)
#endif
    for (std::int64_t qi64 = 0; qi64 < static_cast<std::int64_t>(nq); ++qi64) {
      const std::size_t qi = static_cast<std::size_t>(qi64);
      const float* qsrc = queries.data() + qi * dim;
      l2_normalize_into(qsrc, q_unit.data(), dim);
      std::copy(q_unit.begin(), q_unit.end(), q_rot.begin());
      rotation.apply(q_rot);
      project_landmarks(q_rot.data(), dim, landmark_basis, n_landmarks, q_land.data());

      std::size_t heap_sz = 0;
      float heap_min = 0.f;
      landmark_shortlist_scan(q_land, n_landmarks, blocked_landmarks, n_vectors, n_blocks, sl,
                              heap_s.data(), heap_i.data(), heap_sz, heap_min);

      for (std::size_t i = 0; i < heap_sz; ++i) {
        cand[i] = heap_i[i];
      }
      exact_rerank_topk(q_unit, dim, db_unit, std::span<const std::uint64_t>(cand.data(), heap_sz),
                        heap_sz, k, out.scores.data() + qi * k, out.ids.data() + qi * k);
    }
  }

  return out;
}

std::string search_backend_name(std::size_t, std::size_t) {
#if VECTORCACHE_LANDMARK_AVX2 || defined(__AVX2__) || defined(_MSC_VER)
  return "landmark-avx2+rerank";
#else
  return "landmark-scalar+rerank";
#endif
}

std::string search_backend_diagnostics(std::size_t, std::size_t) {
  return std::string("backend=") + search_backend_name(0, 0) +
         " algo=fwht1->4bit->landmarks->exact_rerank\n";
}

}  // namespace vectorcache
