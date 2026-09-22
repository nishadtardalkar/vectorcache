#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
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
          cc.split_bucket(j, vectors, keys);
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
  double energy = 0.0;
  for (float v : mean) {
    energy += static_cast<double>(v) * static_cast<double>(v);
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(energy));
  for (float& v : mean) {
    v *= inv;
  }
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
    std::vector<float> v(dim, 0.0f);
    v[0] = 1.0f;
    v[1] = 0.05f * static_cast<float>(static_cast<int>(i) - 3);
    double energy = 0.0;
    for (float x : v) {
      energy += static_cast<double>(x) * static_cast<double>(x);
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(energy));
    for (float& x : v) {
      x *= inv;
    }
    keys.push_back(cc.assign_and_update(v));
    vectors.insert(vectors.end(), v.begin(), v.end());
  }

  EXPECT_EQ(cc.num_buckets(), 1u);
  EXPECT_EQ(cc.count(0), n);
  cc.split_bucket(0, vectors, keys);

  EXPECT_EQ(cc.num_buckets(), 2u);
  const std::size_t n0 = cc.count(0);
  const std::size_t n1 = cc.count(1);
  EXPECT_EQ(n0 + n1, n);
  EXPECT_LE(n0 > n1 ? n0 - n1 : n1 - n0, 1u);
  EXPECT_NEAR(std::sqrt(dot(cc.centroid(0), cc.centroid(0))), 1.0f, 1e-5f);
  EXPECT_NEAR(std::sqrt(dot(cc.centroid(1), cc.centroid(1))), 1.0f, 1e-5f);
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
      cc.split_bucket(static_cast<std::size_t>(keys.back()), vectors, keys);
    }
  }
  // Force splits until every cell has count <= 1.
  bool split = true;
  while (split) {
    split = false;
    for (std::size_t j = 0; j < cc.num_buckets(); ++j) {
      if (cc.count(j) > 1) {
        cc.split_bucket(j, vectors, keys);
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
            cc.split_bucket(j, vectors, keys);
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
