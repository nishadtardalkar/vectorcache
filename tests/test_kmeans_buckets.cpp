#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

#include "vectorcache/cluster/kmeans_buckets.hpp"
#include "vectorcache/cluster/score_centroids.hpp"
#include "vectorcache/index.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace {

std::vector<float> random_matrix(std::size_t n, std::size_t dim, std::uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> out(n * dim);
  for (float& x : out) x = dist(rng);
  return out;
}

}  // namespace

TEST(ScoreCentroids, AssignNearest) {
  constexpr std::size_t dim = 8;
  std::vector<float> centroids = {
      1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,  // c0
      0.f, 1.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,  // c1
  };
  std::vector<float> rows = {
      0.9f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,  // → 0
      0.1f, 0.9f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f,  // → 1
  };
  vectorcache::normalize_rows_inplace(rows, 2, dim);
  vectorcache::normalize_rows_inplace(centroids, 2, dim);
  std::vector<std::uint32_t> assign(2);
  vectorcache::assign_nearest_centroid(rows, 2, dim, centroids, 2, assign);
  EXPECT_EQ(assign[0], 0u);
  EXPECT_EQ(assign[1], 1u);
}

TEST(KMeansBuckets, AddSearchAndIds) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 512;
  constexpr std::size_t nq = 4;
  constexpr std::size_t k = 5;

  auto db = random_matrix(n, dim, 7);
  auto queries = random_matrix(nq, dim, 11);

  vectorcache::BucketParams params;
  params.scan_fraction = 1.f;  // open all buckets → match flat coverage
  params.var_threshold = 0.3f;
  params.min_split_size = 64;
  params.split_iters = 5;

  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  index.prepare();
  EXPECT_EQ(index.size(), n);
  EXPECT_GE(index.num_buckets(), 1u);

  auto res = index.search(queries, k);
  ASSERT_EQ(res.nq, nq);
  ASSERT_EQ(res.k, k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_LT(res.ids[qi * k + j], n);
    }
    for (std::size_t j = 1; j < k; ++j) {
      EXPECT_GE(res.scores[qi * k + j - 1], res.scores[qi * k + j]);
    }
  }
}

TEST(KMeansBuckets, VarianceSplitGrowsBuckets) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 400;

  // Two well-separated clusters so variance force-splits.
  std::vector<float> db(n * dim, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    db[i * dim + 0] = (i < n / 2) ? 1.f : 0.f;
    db[i * dim + 1] = (i < n / 2) ? 0.f : 1.f;
    // Small noise.
    db[i * dim + 2] = 0.01f * static_cast<float>((i % 7) - 3);
  }

  vectorcache::BucketParams params;
  params.scan_fraction = 0.5f;
  params.var_threshold = 0.2f;
  params.min_split_size = 32;
  params.split_iters = 8;

  vectorcache::BucketedTurboQuantIndex index(dim, 2, params);
  index.add(db);
  EXPECT_GT(index.num_buckets(), 1u);
  std::size_t total = 0;
  for (std::size_t b = 0; b < index.num_buckets(); ++b) {
    total += index.bucket_size(b);
  }
  EXPECT_EQ(total, n);
}

TEST(KMeansBuckets, MaxBucketSizeForcesSplit) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 800;

  // Tight cluster: variance stays low, so only max_bucket_size should force splits.
  auto db = random_matrix(n, dim, 42);
  for (std::size_t i = 0; i < n; ++i) {
    db[i * dim + 0] += 10.f;  // pull toward same direction
  }

  vectorcache::BucketParams params;
  params.scan_fraction = 0.2f;
  params.var_threshold = 10.f;  // effectively disable variance splits
  params.min_split_size = 32;
  params.max_bucket_size = 64;
  params.split_iters = 5;

  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  EXPECT_GT(index.num_buckets(), 1u);
  for (std::size_t b = 0; b < index.num_buckets(); ++b) {
    EXPECT_LE(index.bucket_size(b), params.max_bucket_size);
  }
}

TEST(KMeansBuckets, ScanFractionOpensSubset) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 300;

  auto db = random_matrix(n, dim, 99);
  vectorcache::BucketParams params;
  params.scan_fraction = 0.2f;
  params.var_threshold = 0.25f;
  params.min_split_size = 40;

  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  index.prepare();
  ASSERT_GE(index.num_buckets(), 1u);

  auto q = random_matrix(1, dim, 123);
  auto res = index.search(q, 3);
  EXPECT_EQ(res.k, 3u);
  EXPECT_LT(res.ids[0], n);
}

TEST(KMeansBuckets, FullScanMatchesFlatIdsOften) {
  // With scan_fraction=1, bucketed search scores the same vectors as flat (partitioned).
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 256;
  constexpr std::size_t k = 5;

  auto db = random_matrix(n, dim, 3);
  auto q = std::span<const float>(db.data(), dim);

  vectorcache::TurboQuantIndex flat(dim, 4);
  flat.add(db);
  flat.prepare();
  auto flat_res = flat.search(q, k);

  vectorcache::BucketParams params;
  params.scan_fraction = 1.f;
  params.var_threshold = 0.4f;
  params.min_split_size = 64;

  vectorcache::BucketedTurboQuantIndex bucketed(dim, 4, params);
  bucketed.add(db);
  bucketed.prepare();
  auto b_res = bucketed.search(q, k);

  ASSERT_EQ(flat_res.k, b_res.k);
  // Top-1 should usually agree when scanning everything; allow rare quantization ties.
  EXPECT_EQ(flat_res.ids[0], b_res.ids[0]);
}

TEST(KMeansBuckets, SharedHeapFullScanK64Deterministic) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 512;
  constexpr std::size_t k = 64;
  constexpr std::size_t nq = 4;

  auto db = random_matrix(n, dim, 11);
  auto queries = random_matrix(nq, dim, 12);

  vectorcache::BucketParams params;
  params.scan_fraction = 1.f;
  params.var_threshold = 0.3f;
  params.min_split_size = 64;
  params.max_bucket_size = 128;

  vectorcache::BucketedTurboQuantIndex a(dim, 4, params);
  a.add(db);
  a.prepare();
  ASSERT_GT(a.num_buckets(), 1u);
  auto r1 = a.search(queries, k);
  auto r2 = a.search(queries, k);

  ASSERT_EQ(r1.nq, nq);
  ASSERT_EQ(r1.k, k);
  ASSERT_EQ(r1.ids, r2.ids);
  ASSERT_EQ(r1.scores, r2.scores);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_LT(r1.ids[qi * k + j], n);
      if (j + 1 < k) {
        EXPECT_GE(r1.scores[qi * k + j], r1.scores[qi * k + j + 1]);
      }
    }
  }
}

TEST(SearchInto, ContinuesAcrossRangesMatchesSinglePass) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 256;
  constexpr std::size_t k = 64;
  constexpr std::size_t nq = 2;
  constexpr std::size_t bits = 4;

  auto db = random_matrix(n, dim, 21);
  auto queries = random_matrix(nq, dim, 22);

  vectorcache::TurboQuantIndex index(dim, bits);
  index.add(db);
  index.prepare();
  auto once = index.search(queries, k);

  auto cb = vectorcache::codebook(bits, dim);
  vectorcache::Rotation rotation(dim);
  auto [blocked, n_blocks] = vectorcache::repack(index.packed_codes(), n, bits, dim);
  auto prep = vectorcache::prepare_queries(queries, nq, dim, rotation, cb.second, bits, {}, {});

  const std::size_t mid_blocks = std::max<std::size_t>(1, n_blocks / 2);
  const std::size_t n_byte_groups = dim / (8 / bits);
  const std::size_t block_bytes = n_byte_groups * vectorcache::kBlock;
  const std::size_t mid_vecs = std::min(n, mid_blocks * vectorcache::kBlock);
  const std::size_t rest_vecs = n - mid_vecs;
  const std::size_t rest_blocks = n_blocks - mid_blocks;

  std::vector<std::uint64_t> ids0(mid_vecs), ids1(rest_vecs);
  for (std::size_t i = 0; i < mid_vecs; ++i) ids0[i] = i;
  for (std::size_t i = 0; i < rest_vecs; ++i) ids1[i] = mid_vecs + i;

  std::vector<float> heap_s(nq * k, 0.f);
  std::vector<std::uint64_t> heap_i(nq * k, 0);
  std::vector<std::size_t> heap_sz(nq, 0);
  std::vector<float> heap_min(nq, 0.f);
  std::vector<std::size_t> heap_mi(nq, 0);
  std::vector<std::size_t> all_q(nq);
  std::iota(all_q.begin(), all_q.end(), 0);

  auto scales = index.scales();
  score_prepared_into(prep, k, std::span<const std::uint8_t>(blocked.data(), mid_blocks * block_bytes),
                      mid_blocks, scales.subspan(0, mid_vecs), ids0, all_q, heap_s.data(),
                      heap_i.data(), heap_sz.data(), heap_min.data(), heap_mi.data());
  if (rest_blocks > 0 && rest_vecs > 0) {
    score_prepared_into(
        prep, k,
        std::span<const std::uint8_t>(blocked.data() + mid_blocks * block_bytes,
                                      rest_blocks * block_bytes),
        rest_blocks, scales.subspan(mid_vecs), ids1, all_q, heap_s.data(), heap_i.data(),
        heap_sz.data(), heap_min.data(), heap_mi.data());
  }

  vectorcache::SearchResults into_res;
  vectorcache::heaps_to_search_results(into_res, nq, k, heap_s.data(), heap_i.data(),
                                       heap_sz.data());
  ASSERT_EQ(once.k, into_res.k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    EXPECT_EQ(once.ids[qi * k], into_res.ids[qi * k]) << "query " << qi;
    std::vector<std::uint64_t> a(once.ids.begin() + static_cast<std::ptrdiff_t>(qi * k),
                                 once.ids.begin() + static_cast<std::ptrdiff_t>((qi + 1) * k));
    std::vector<std::uint64_t> b(into_res.ids.begin() + static_cast<std::ptrdiff_t>(qi * k),
                                 into_res.ids.begin() + static_cast<std::ptrdiff_t>((qi + 1) * k));
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    EXPECT_EQ(a, b) << "query " << qi;
  }
}
