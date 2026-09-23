#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vectorcache/cluster/kmeans_buckets.hpp"
#include "vectorcache/cluster/score_centroids.hpp"
#include "vectorcache/index.hpp"

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
