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

vectorcache::BucketParams test_params(float scan_fraction = 1.f, std::size_t min_sz = 256,
                                      std::size_t max_sz = 800, float cos_var = 0.02f) {
  vectorcache::BucketParams params;
  params.scan_fraction = scan_fraction;
  params.min_bucket_size = min_sz;
  params.max_bucket_size = max_sz;
  params.cos_var_threshold = cos_var;
  // Keep pulse schedule off-scale for unit tests unless a test opts in.
  params.energy_soft = (std::max)(max_sz, min_sz);
  params.pulse_soft = (std::max)(max_sz, min_sz);
  params.expected_n = 0;
  return params;
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

  // Small max so random data splits; full scan for coverage.
  auto params = test_params(1.f, /*min*/ 8, /*max*/ 64, /*var*/ 0.02f);

  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  index.prepare();
  EXPECT_EQ(index.size(), n);
  EXPECT_GE(index.num_buckets(), 1u);
  for (std::size_t b = 0; b < index.num_buckets(); ++b) {
    EXPECT_LE(index.bucket_size(b), params.max_bucket_size);
  }

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

TEST(KMeansBuckets, SameDirectionStaysOneClusterBelowMax) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 50;

  // Nearly parallel → low cosine variance; under max → single cluster.
  std::vector<float> db(n * dim, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    db[i * dim + 0] = 1.f;
    db[i * dim + 1] = 0.01f * static_cast<float>(static_cast<int>(i % 5) - 2);
  }

  auto params = test_params(1.f, /*min*/ 8, /*max*/ 2048, /*var*/ 0.02f);
  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  EXPECT_EQ(index.num_buckets(), 1u);
  EXPECT_EQ(index.bucket_size(0), n);
}

TEST(KMeansBuckets, RoutingMeanMovesAfterJoins) {
  constexpr std::size_t dim = 16;

  std::vector<float> first(dim, 0.f);
  first[0] = 1.f;

  std::vector<float> joiners(3 * dim, 0.f);
  for (std::size_t i = 0; i < 3; ++i) {
    joiners[i * dim + 0] = 1.f;
    joiners[i * dim + 1] = 0.1f * static_cast<float>(i + 1);
  }

  auto params = test_params(1.f, /*min*/ 8, /*max*/ 2048, /*var*/ 0.02f);
  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(first);
  ASSERT_EQ(index.num_buckets(), 1u);
  auto c0 = index.bucket_centroid(0);
  std::vector<float> founding_routing(c0.begin(), c0.end());

  index.add(joiners);
  EXPECT_EQ(index.num_buckets(), 1u);
  EXPECT_EQ(index.bucket_size(0), 4u);
  auto c1 = index.bucket_centroid(0);
  ASSERT_EQ(c1.size(), founding_routing.size());

  double diff = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    diff += std::abs(static_cast<double>(c1[d]) - static_cast<double>(founding_routing[d]));
  }
  EXPECT_GT(diff, 1e-4) << "routing centroid should move toward running mean";
}

TEST(KMeansBuckets, EncodeCentroidFrozenWhileRoutingMoves) {
  constexpr std::size_t dim = 16;

  std::vector<float> first(dim, 0.f);
  first[0] = 1.f;

  std::vector<float> joiners(3 * dim, 0.f);
  for (std::size_t i = 0; i < 3; ++i) {
    joiners[i * dim + 0] = 1.f;
    joiners[i * dim + 1] = 0.1f * static_cast<float>(i + 1);
  }

  auto params = test_params(1.f, /*min*/ 8, /*max*/ 2048, /*var*/ 0.02f);
  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(first);
  ASSERT_EQ(index.num_buckets(), 1u);
  auto enc0 = index.bucket_encode_centroid(0);
  std::vector<float> frozen(enc0.begin(), enc0.end());

  index.add(joiners);
  EXPECT_EQ(index.num_buckets(), 1u);
  auto enc1 = index.bucket_encode_centroid(0);
  ASSERT_EQ(enc1.size(), frozen.size());
  for (std::size_t d = 0; d < dim; ++d) {
    EXPECT_FLOAT_EQ(enc1[d], frozen[d]);
  }

  // Routing moved, encode did not.
  auto route = index.bucket_centroid(0);
  double route_diff = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    route_diff += std::abs(static_cast<double>(route[d]) - static_cast<double>(frozen[d]));
  }
  EXPECT_GT(route_diff, 1e-4);
}

TEST(KMeansBuckets, MaxBucketSizeForcesSplit) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 20;
  constexpr std::size_t max_sz = 4;

  // Near-duplicates → tiny cosine variance; max size still forces splits.
  std::vector<float> db(n * dim, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    db[i * dim + 0] = 1.f;
    db[i * dim + 1] = 1e-4f * static_cast<float>(i);
  }

  auto params = test_params(1.f, /*min*/ 2, /*max*/ max_sz, /*var*/ 1.0f);  // var never triggers
  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  EXPECT_GT(index.num_buckets(), 1u);
  for (std::size_t b = 0; b < index.num_buckets(); ++b) {
    EXPECT_LE(index.bucket_size(b), max_sz);
  }
  std::size_t total = 0;
  for (std::size_t b = 0; b < index.num_buckets(); ++b) total += index.bucket_size(b);
  EXPECT_EQ(total, n);
}

TEST(KMeansBuckets, EnergyOrHardSplitOnDiverseVectors) {
  constexpr std::size_t dim = 32;
  constexpr std::size_t n = 64;

  // Mix of orthogonal axes → energy / hard fission creates multiple lists.
  std::vector<float> db(n * dim, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t axis = i % 8;
    db[i * dim + axis] = 1.f;
  }

  auto params = test_params(1.f, /*min*/ 4, /*max*/ 32, /*var*/ 0.01f);
  params.energy_soft = 8;
  params.energy_trig = 0.01f;
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

  std::vector<float> db(n * dim, 0.f);
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t axis = i % 8;
    db[i * dim + axis] = 1.f;
    db[i * dim + ((axis + 1) % dim)] = 0.02f * static_cast<float>(static_cast<int>(i % 5) - 2);
  }

  auto params = test_params(0.2f, /*min*/ 4, /*max*/ 40, /*var*/ 0.01f);
  vectorcache::BucketedTurboQuantIndex index(dim, 4, params);
  index.add(db);
  index.prepare();
  ASSERT_GE(index.num_buckets(), 2u);

  auto q = random_matrix(1, dim, 123);
  auto res = index.search(q, 3);
  EXPECT_EQ(res.k, 3u);
  EXPECT_LT(res.ids[0], n);
}

TEST(KMeansBuckets, FullScanReturnsValidTopK) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 256;
  constexpr std::size_t k = 5;

  auto db = random_matrix(n, dim, 3);
  auto q = std::span<const float>(db.data(), dim);

  auto params = test_params(1.f, /*min*/ 8, /*max*/ 64, /*var*/ 0.02f);
  vectorcache::BucketedTurboQuantIndex bucketed(dim, 4, params);
  bucketed.add(db);
  bucketed.prepare();
  auto b_res = bucketed.search(q, k);

  ASSERT_EQ(b_res.k, k);
  EXPECT_LT(b_res.ids[0], n);
  for (std::size_t j = 1; j < k; ++j) {
    EXPECT_GE(b_res.scores[j - 1], b_res.scores[j]);
    EXPECT_LT(b_res.ids[j], n);
  }
}

TEST(KMeansBuckets, SharedHeapFullScanK64Deterministic) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 512;
  constexpr std::size_t k = 64;
  constexpr std::size_t nq = 4;

  auto db = random_matrix(n, dim, 11);
  auto queries = random_matrix(nq, dim, 12);

  auto params = test_params(1.f, /*min*/ 8, /*max*/ 64, /*var*/ 0.02f);
  vectorcache::BucketedTurboQuantIndex a(dim, 4, params);
  a.add(db);
  a.prepare();
  ASSERT_GE(a.num_buckets(), 1u);
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
