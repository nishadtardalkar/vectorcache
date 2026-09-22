#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/engine.hpp"

using namespace vectorcache;

namespace {

class MockReader : public datasets::DatasetReader {
 public:
  MockReader(std::vector<std::vector<float>> vectors, std::size_t dim)
      : vectors_(std::move(vectors)), dim_(dim) {}

  datasets::DatasetMeta meta() const override {
    return {dim_, vectors_.size(), "mock"};
  }

  bool next_vector_into(std::span<float> out) override {
    if (index_ >= vectors_.size()) {
      return false;
    }
    const auto& v = vectors_[index_++];
    std::copy(v.begin(), v.end(), out.begin());
    return true;
  }

 private:
  std::vector<std::vector<float>> vectors_;
  std::size_t dim_;
  std::size_t index_ = 0;
};

float dot(std::span<const float> a, std::span<const float> b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    s += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(s);
}

std::vector<float> unit_axis(std::size_t dim, std::size_t axis) {
  std::vector<float> v(dim, 0.0f);
  v[axis % dim] = 1.0f;
  return v;
}

/// L2-normalize a vector in place (assumes non-zero).
void normalize_inplace(std::vector<float>& v) {
  double energy = 0.0;
  for (float x : v) {
    energy += static_cast<double>(x) * static_cast<double>(x);
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(energy));
  for (float& x : v) {
    x *= inv;
  }
}

std::vector<float> unit2(std::size_t dim, float a0, float a1) {
  std::vector<float> v(dim, 0.0f);
  v[0] = a0;
  v[1] = a1;
  normalize_inplace(v);
  return v;
}

/// Grow centroids to at least `min_buckets` by assigning distinct axes and splitting at max=1.
void grow_buckets(index::ClusterCentroids& cc, std::size_t min_buckets,
                  std::vector<float>& vectors, std::vector<std::uint64_t>& keys) {
  const std::size_t dim = cc.dim();
  std::size_t axis = 0;
  while (cc.num_buckets() < min_buckets) {
    auto v = unit_axis(dim, axis++);
    keys.push_back(cc.assign_and_update(v));
    vectors.insert(vectors.end(), v.begin(), v.end());
    bool split = true;
    while (split) {
      split = false;
      for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
        if (cc.count(j) > 1) {
          cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
          split = true;
          break;
        }
      }
    }
    if (axis > min_buckets * 8) {
      break;
    }
  }
}

}  // namespace

TEST(ClusterBucketsTest, StartsEmptyThenSeedsFirst) {
  const std::size_t dim = 16;
  index::ClusterCentroids cc(dim);
  EXPECT_TRUE(cc.empty());
  EXPECT_EQ(cc.num_buckets(), 0u);

  auto v = unit_axis(dim, 0);
  const auto key = cc.assign_and_update(v);
  EXPECT_EQ(key, 0u);
  EXPECT_EQ(cc.num_buckets(), 1u);
  EXPECT_EQ(cc.count(0), 1u);
  EXPECT_NEAR(dot(cc.centroid(0), v), 1.0f, 1e-5f);
}

TEST(ClusterBucketsTest, AssignAndUpdateMatchesBatchMean) {
  const std::size_t dim = 8;
  index::ClusterCentroids cc(dim);

  std::vector<float> x0 = unit_axis(dim, 0);
  std::vector<float> x1 = unit_axis(dim, 0);
  x1[0] = 0.8f;
  x1[1] = 0.6f;

  const std::uint64_t k0 = cc.assign_and_update(x0);
  const std::uint64_t k1 = cc.assign_and_update(x1);
  EXPECT_EQ(k0, k1);

  const std::size_t j = static_cast<std::size_t>(k0);
  EXPECT_EQ(cc.count(j), 2u);

  std::vector<float> mean(dim, 0.0f);
  for (std::size_t d = 0; d < dim; ++d) {
    mean[d] = 0.5f * (x0[d] + x1[d]);
  }
  normalize_inplace(mean);
  EXPECT_NEAR(dot(cc.centroid(j), mean), 1.0f, 1e-5f);
}

TEST(ClusterBucketsTest, SplitMedianBalanced) {
  const std::size_t dim = 8;
  index::ClusterCentroids cc(dim);
  std::vector<float> vectors;
  std::vector<std::uint64_t> keys;

  // Near-duplicates along axis 0 (with a slight axis-1 tilt so A/B are distinct).
  const std::size_t n = 8;
  for (std::size_t i = 0; i < n; ++i) {
    auto v = unit2(dim, 1.0f, 0.05f * static_cast<float>(static_cast<int>(i) - 3));
    keys.push_back(cc.assign_and_update(v));
    vectors.insert(vectors.end(), v.begin(), v.end());
  }

  EXPECT_EQ(cc.num_buckets(), 1u);
  EXPECT_EQ(cc.count(0), n);
  // Median cut + global eject (B=2 → sibling-only); near-line geometry stays ~balanced.
  cc.split_bucket(0, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);

  EXPECT_EQ(cc.num_buckets(), 2u);
  const std::size_t n0 = cc.count(0);
  const std::size_t n1 = cc.count(1);
  EXPECT_EQ(n0 + n1, n);
  EXPECT_GE(n0, 1u);
  EXPECT_GE(n1, 1u);
  EXPECT_NEAR(std::sqrt(dot(cc.centroid(0), cc.centroid(0))), 1.0f, 1e-5f);
  EXPECT_NEAR(std::sqrt(dot(cc.centroid(1), cc.centroid(1))), 1.0f, 1e-5f);
}

TEST(ClusterBucketsTest, SplitLocalVoronoiReassigns) {
  const std::size_t dim = 8;
  // Asymmetric tilts: after a median cut, the near-cut point on the sparse side is closer
  // to the dense-side child centroid; Lloyd and/or global eject should move it.
  const std::vector<float> tilts = {-1.0f, -0.95f, -0.9f, -0.05f, 0.2f, 0.25f, 0.3f, 0.35f};

  auto run_split = [&](std::size_t lloyd) {
    index::ClusterCentroids cc(dim);
    std::vector<float> vectors;
    std::vector<std::uint64_t> keys;
    for (float t : tilts) {
      auto v = unit2(dim, 1.0f, t);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }
    cc.split_bucket(0, vectors, keys, lloyd, /*steal_neighbors=*/0);
    return std::make_tuple(std::move(cc), std::move(vectors), std::move(keys));
  };

  auto [cc0, vec0, keys0] = run_split(0);
  auto [cc1, vec1, keys1] = run_split(1);
  ASSERT_EQ(cc0.num_buckets(), 2u);
  ASSERT_EQ(cc1.num_buckets(), 2u);

  // Global eject (lloyd=0) and Lloyd both leave members preferring their child among {0,1}.
  for (std::size_t i = 0; i < keys0.size(); ++i) {
    const std::span<const float> x(vec0.data() + i * dim, dim);
    const std::size_t k = static_cast<std::size_t>(keys0[i]);
    const std::size_t other = k == 0 ? 1 : 0;
    EXPECT_GE(dot(cc0.centroid(k), x), dot(cc0.centroid(other), x));
  }
  for (std::size_t i = 0; i < keys1.size(); ++i) {
    const std::span<const float> x(vec1.data() + i * dim, dim);
    const std::size_t k = static_cast<std::size_t>(keys1[i]);
    const std::size_t other = k == 0 ? 1 : 0;
    EXPECT_GE(dot(cc1.centroid(k), x), dot(cc1.centroid(other), x));
  }
}

TEST(ClusterBucketsTest, SplitNeighborStealMovesBoundary) {
  const std::size_t dim = 4;

  auto run = [&](std::size_t steal, std::uint64_t* out_bkey, std::uint64_t* out_after) {
    index::ClusterCentroids cc(dim);
    std::vector<float> vectors;
    std::vector<std::uint64_t> keys;

    for (std::size_t a = 0; a < 3; ++a) {
      auto v = unit_axis(dim, a);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
      for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
        if (cc.count(j) > 1) {
          cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
          break;
        }
      }
    }
    if (cc.num_buckets() != 3) {
      *out_bkey = 0;
      *out_after = 0;
      return false;
    }

    auto find_axis = [&](std::size_t axis) {
      auto ax = unit_axis(dim, axis);
      std::size_t best = 0;
      float best_ip = -2.0f;
      for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
        const float ip = dot(cc.centroid(j), ax);
        if (ip > best_ip) {
          best_ip = ip;
          best = j;
        }
      }
      return best;
    };
    const std::size_t e0 = find_axis(0);
    const std::size_t e1 = find_axis(1);
    if (e0 == e1) {
      return false;
    }

    // Anchor e1 near the axis so a later boundary point cannot dominate ĉ_e1.
    for (int i = 0; i < 6; ++i) {
      auto v = unit_axis(dim, 1);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }

    // Boundary leans e1 (lives in e1) but is closer to a +tilt e0-child after split.
    auto boundary = unit2(dim, 0.90f, 0.95f);
    keys.push_back(cc.assign_and_update(boundary));
    vectors.insert(vectors.end(), boundary.begin(), boundary.end());
    const std::size_t bidx = keys.size() - 1;
    if (keys[bidx] != static_cast<std::uint64_t>(e1)) {
      return false;
    }

    // Overcrowd e0 along a diametral tilt; positive half becomes the attractor.
    for (int i = 0; i < 12; ++i) {
      const float t = -0.5f + 0.1f * static_cast<float>(i);
      auto v = unit2(dim, 1.0f, t);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }
    if (cc.count(e0) <= 2 || keys[bidx] != static_cast<std::uint64_t>(e1)) {
      return false;
    }

    cc.split_bucket(e0, vectors, keys, /*lloyd_iters=*/1, steal);
    *out_bkey = static_cast<std::uint64_t>(e1);
    *out_after = keys[bidx];
    return true;
  };

  std::uint64_t bkey0 = 0;
  std::uint64_t after0 = 0;
  ASSERT_TRUE(run(/*steal=*/0, &bkey0, &after0));
  EXPECT_EQ(after0, bkey0);

  std::uint64_t bkey1 = 0;
  std::uint64_t after1 = 0;
  ASSERT_TRUE(run(/*steal=*/4, &bkey1, &after1));
  EXPECT_NE(after1, bkey1);
}

TEST(ClusterBucketsTest, SplitGlobalEjectMovesFromChildren) {
  const std::size_t dim = 4;

  auto seed_axes = [&](index::ClusterCentroids& cc, std::vector<float>& vectors,
                       std::vector<std::uint64_t>& keys) {
    for (std::size_t a = 0; a < 3; ++a) {
      auto v = unit_axis(dim, a);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
      for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
        if (cc.count(j) > 1) {
          cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
          break;
        }
      }
    }
  };

  auto find_axis = [&](const index::ClusterCentroids& cc, std::size_t axis) {
    auto ax = unit_axis(dim, axis);
    std::size_t best = 0;
    float best_ip = -2.0f;
    for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
      const float ip = dot(cc.centroid(j), ax);
      if (ip > best_ip) {
        best_ip = ip;
        best = j;
      }
    }
    return best;
  };

  // Pre-tilt e0 so a strong e1-leaning outlier still inserts into e0; dilute with pure
  // axis0 so global nearest becomes e1 while the sticky key stays e0; split then ejects.
  // Diametral A is the outlier (far from axis0 ĉ) → lands on median side 0 (j).
  {
    index::ClusterCentroids cc(dim);
    std::vector<float> vectors;
    std::vector<std::uint64_t> keys;
    seed_axes(cc, vectors, keys);
    ASSERT_EQ(cc.num_buckets(), 3u);
    const std::size_t e0 = find_axis(cc, 0);
    const std::size_t e1 = find_axis(cc, 1);
    ASSERT_NE(e0, e1);

    // Pre-tilt stays on e0 (x0-dominant) so the later e1-leaning outlier still inserts there.
    for (int i = 0; i < 10; ++i) {
      auto v = unit2(dim, 0.75f, 0.66f);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }

    auto outlier = unit2(dim, 0.50f, 0.87f);
    keys.push_back(cc.assign_and_update(outlier));
    vectors.insert(vectors.end(), outlier.begin(), outlier.end());
    const std::size_t oidx = keys.size() - 1;
    ASSERT_EQ(keys[oidx], static_cast<std::uint64_t>(e0));

    for (int i = 0; i < 24; ++i) {
      auto v = unit_axis(dim, 0);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }
    ASSERT_EQ(keys[oidx], static_cast<std::uint64_t>(e0));
    ASSERT_EQ(static_cast<std::size_t>(cc.nearest(outlier)), e1);

    const std::size_t j = e0;
    cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
    EXPECT_EQ(keys[oidx], static_cast<std::uint64_t>(e1));
    EXPECT_EQ(cc.nearest(outlier), keys[oidx]);
  }

  // Same sticky setup, but add an opposite extreme so the outlier becomes diametral B
  // (high s → median side 1 / j_new) while still diluted enough for e1 to win.
  {
    index::ClusterCentroids cc(dim);
    std::vector<float> vectors;
    std::vector<std::uint64_t> keys;
    seed_axes(cc, vectors, keys);
    ASSERT_EQ(cc.num_buckets(), 3u);
    const std::size_t e0 = find_axis(cc, 0);
    const std::size_t e1 = find_axis(cc, 1);
    ASSERT_NE(e0, e1);

    for (int i = 0; i < 10; ++i) {
      auto v = unit2(dim, 0.75f, 0.66f);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }

    auto outlier = unit2(dim, 0.50f, 0.87f);
    keys.push_back(cc.assign_and_update(outlier));
    vectors.insert(vectors.end(), outlier.begin(), outlier.end());
    const std::size_t oidx = keys.size() - 1;
    ASSERT_EQ(keys[oidx], static_cast<std::uint64_t>(e0));

    // Opposite extreme on axis1 so A/B span negatives ↔ outlier; outlier tends to B / j_new.
    for (int i = 0; i < 6; ++i) {
      auto v = unit2(dim, 0.75f, -0.66f);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }

    for (int i = 0; i < 24; ++i) {
      auto v = unit_axis(dim, 0);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
    }
    ASSERT_EQ(keys[oidx], static_cast<std::uint64_t>(e0));
    ASSERT_EQ(static_cast<std::size_t>(cc.nearest(outlier)), e1);

    const std::size_t j_before = e0;
    const std::size_t b_before = cc.num_buckets();
    cc.split_bucket(j_before, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
    ASSERT_EQ(cc.num_buckets(), b_before + 1);
    // Landed on the new child before eject, then moved to e1.
    EXPECT_EQ(keys[oidx], static_cast<std::uint64_t>(e1));
    EXPECT_EQ(cc.nearest(outlier), keys[oidx]);
  }
}

TEST(ClusterBucketsTest, SplitGrowsAndCapsCount) {
  const std::size_t dim = 4;
  index::ClusterCentroids cc(dim);
  std::vector<float> vectors;
  std::vector<std::uint64_t> keys;

  // Three near-orthogonal vectors → after overflows, expect growth.
  for (std::size_t a = 0; a < 3; ++a) {
    auto v = unit_axis(dim, a);
    keys.push_back(cc.assign_and_update(v));
    vectors.insert(vectors.end(), v.begin(), v.end());
    if (cc.count(static_cast<std::size_t>(keys.back())) > 1) {
      cc.split_bucket(static_cast<std::size_t>(keys.back()), vectors, keys,
                      /*lloyd_iters=*/0, /*steal_neighbors=*/0);
    }
  }
  // Force splits until every cell has count <= 1.
  bool split = true;
  while (split) {
    split = false;
    for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
      if (cc.count(j) > 1) {
        cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
        split = true;
        break;
      }
    }
  }

  EXPECT_GE(cc.num_buckets(), 2u);
  for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
    EXPECT_LE(cc.count(j), 1u);
    EXPECT_NEAR(std::sqrt(dot(cc.centroid(j), cc.centroid(j))), 1.0f, 1e-5f);
  }
  EXPECT_EQ(keys.size(), 3u);
}

TEST(ClusterBucketsTest, ProbeCoverageStopsAtFraction) {
  const std::size_t dim = 4;
  index::ClusterCentroids cc(dim);
  std::vector<float> vectors;
  std::vector<std::uint64_t> keys;

  // Uneven membership: 4 on axis0, 2 on axis1, 1 on axis2 (N=7), with splits at max=2.
  auto ingest_axis = [&](std::size_t axis, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) {
      auto v = unit_axis(dim, axis);
      keys.push_back(cc.assign_and_update(v));
      vectors.insert(vectors.end(), v.begin(), v.end());
      bool split = true;
      while (split) {
        split = false;
        for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
          if (cc.count(j) > 2) {
            cc.split_bucket(j, vectors, keys, /*lloyd_iters=*/0, /*steal_neighbors=*/0);
            split = true;
            break;
          }
        }
      }
    }
  };
  ingest_axis(0, 4);
  ingest_axis(1, 2);
  ingest_axis(2, 1);

  std::vector<std::uint64_t> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  auto idx = index::BucketIndex::build(sorted, std::move(cc));
  ASSERT_EQ(idx.size(), 7u);

  auto q = unit_axis(dim, 0);
  std::size_t candidates = 0;
  const auto half = idx.probe(q, 0.5f, &candidates);
  EXPECT_FALSE(half.empty());
  EXPECT_GE(candidates, 4u);

  candidates = 0;
  const auto tiny = idx.probe(q, 1e-6f, &candidates);
  EXPECT_EQ(tiny.size(), 1u);
  EXPECT_GE(candidates, 1u);

  candidates = 0;
  const auto full = idx.probe(q, 1.0f, &candidates);
  EXPECT_EQ(candidates, 7u);

  EXPECT_THROW(index::validate_probe_fraction(0.0f), Error);
  EXPECT_THROW(index::validate_probe_fraction(1.1f), Error);
  EXPECT_THROW(index::validate_probe_fraction(std::numeric_limits<float>::quiet_NaN()), Error);
}

TEST(ClusterBucketsTest, CsrRangesContiguous) {
  const std::size_t dim = 4;
  index::ClusterCentroids cc(dim);
  std::vector<float> vectors;
  std::vector<std::uint64_t> grow_keys;
  grow_buckets(cc, 3, vectors, grow_keys);
  ASSERT_GE(cc.num_buckets(), 3u);

  std::vector<std::uint64_t> keys = {0, 0, 1, 2, 2};
  auto idx = index::BucketIndex::build(keys, std::move(cc));
  EXPECT_EQ(idx.num_cells(), 3u);
  EXPECT_EQ(idx.size(), 5u);
  EXPECT_EQ(idx.cell(0).start, 0u);
  EXPECT_EQ(idx.cell(0).length, 2u);
  EXPECT_EQ(idx.cell(1).start, 2u);
  EXPECT_EQ(idx.cell(1).length, 1u);
  EXPECT_EQ(idx.find(1).length, 1u);
  EXPECT_EQ(idx.find(99).length, 0u);
}

TEST(ClusterBucketsTest, FinalizeBuildsBuckets) {
  const std::size_t dim = 8;
  const std::size_t words = 1;
  ingest::VectorStore store(words, dim, dim, 1);
  std::vector<std::uint64_t> code(words, 0);
  for (std::size_t i = 0; i < 5; ++i) {
    store.push(i, code, 1.0f);
  }
  index::ClusterCentroids cc(dim);
  std::vector<float> vectors;
  std::vector<std::uint64_t> grow_keys;
  grow_buckets(cc, 3, vectors, grow_keys);
  ASSERT_GE(cc.num_buckets(), 3u);

  std::vector<std::uint64_t> keys = {2, 0, 2, 1, 0};
  store.finalize_buckets(keys, std::move(cc));
  EXPECT_TRUE(store.has_buckets());
  EXPECT_EQ(store.buckets().num_cells(), 3u);
  EXPECT_EQ(store.id_at(0), 1u);  // key 0 first after argsort
}

TEST(ClusterBucketsTest, SelfHitWithProbeFraction) {
  const std::size_t dim = 32;
  const std::size_t n = 32;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<float> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = static_cast<float>((i * 17 + j * 3) % 97) / 97.0f;
    }
    vectors.push_back(std::move(v));
  }

  MockReader reader(vectors, dim);
  ingest::BucketParams bp;
  bp.max_bucket_items = 4;
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42, 1, bp);
  ASSERT_EQ(engine.ingest(reader).vectors_ingested, n);
  EXPECT_GE(engine.store().buckets().num_buckets(), 2u);

  auto qe = query::QueryEngine::with_rotation(engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 5;
  params.probe_fraction = 0.25f;
  const auto hits = qe.search(vectors[0], params);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits.front().id, 0u);
}

TEST(ClusterBucketsTest, ProbeSingleBucket) {
  const std::size_t dim = 8;
  index::ClusterCentroids cc(dim);
  auto v = unit_axis(dim, 0);
  const auto key = cc.assign_and_update(v);
  std::vector<std::uint64_t> keys = {key};
  auto idx = index::BucketIndex::build(keys, std::move(cc));

  std::size_t candidates = 0;
  const auto ranges = idx.probe(v, 1.0f, &candidates);
  EXPECT_EQ(ranges.size(), 1u);
  EXPECT_EQ(candidates, 1u);
}
