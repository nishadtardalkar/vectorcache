#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/engine.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"

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

}  // namespace

TEST(ClusterBucketsTest, InitDeterministic) {
  index::ClusterCentroids a(8, 16, 42);
  index::ClusterCentroids b(8, 16, 42);
  index::ClusterCentroids c(8, 16, 43);
  EXPECT_EQ(a.num_buckets(), 8u);
  EXPECT_EQ(a.dim(), 16u);
  for (std::size_t j = 0; j < 8; ++j) {
    EXPECT_FLOAT_EQ(dot(a.centroid(j), b.centroid(j)), 1.0f);
    EXPECT_NEAR(std::sqrt(dot(a.centroid(j), a.centroid(j))), 1.0f, 1e-5f);
  }
  EXPECT_LT(dot(a.centroid(0), c.centroid(0)), 0.999f);
}

TEST(ClusterBucketsTest, AssignAndUpdateMatchesBatchMean) {
  const std::size_t dim = 8;
  index::ClusterCentroids cc(2, dim, 7);

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

TEST(ClusterBucketsTest, RebalanceFixesStickyAssignment) {
  const std::size_t dim = 4;
  const std::size_t B = 2;
  index::ClusterCentroids cc(B, dim, 99);

  // Force both vectors into whatever buckets online assign gives, then shift
  // membership via rebalance with vectors clearly on opposite axes.
  std::vector<float> vectors;
  auto e0 = unit_axis(dim, 0);
  auto e1 = unit_axis(dim, 1);
  vectors.insert(vectors.end(), e0.begin(), e0.end());
  vectors.insert(vectors.end(), e1.begin(), e1.end());

  std::vector<std::uint64_t> keys(2, 0);
  // Pollute: assign both to same online path then rebalance.
  (void)cc.assign_and_update(e0);
  (void)cc.assign_and_update(e0);
  cc.rebalance(vectors, keys);

  EXPECT_NE(keys[0], keys[1]);
  EXPECT_EQ(keys[0], cc.nearest(e0));
  EXPECT_EQ(keys[1], cc.nearest(e1));
  EXPECT_EQ(cc.count(static_cast<std::size_t>(keys[0])), 1u);
  EXPECT_EQ(cc.count(static_cast<std::size_t>(keys[1])), 1u);
}

TEST(ClusterBucketsTest, RebalanceKeepsUnitCentroids) {
  const std::size_t dim = 4;
  const std::size_t B = 2;
  index::ClusterCentroids cc(B, dim, 5);

  std::vector<float> vectors;
  for (std::size_t j = 0; j < B; ++j) {
    const auto c = cc.centroid(j);
    vectors.insert(vectors.end(), c.begin(), c.end());
  }
  std::vector<std::uint64_t> keys(B, 0);

  cc.rebalance(vectors, keys);
  EXPECT_NE(keys[0], keys[1]);
  for (std::size_t j = 0; j < B; ++j) {
    EXPECT_NEAR(std::sqrt(dot(cc.centroid(j), cc.centroid(j))), 1.0f, 1e-5f);
    EXPECT_EQ(cc.count(j), 1u);
  }
}

TEST(ClusterBucketsTest, RebalanceThenAssignKeepsAlignment) {
  const std::size_t dim = 4;
  index::ClusterCentroids cc(2, dim, 11);

  std::vector<float> vectors;
  for (std::size_t j = 0; j < 2; ++j) {
    const auto c = cc.centroid(j);
    vectors.insert(vectors.end(), c.begin(), c.end());
  }
  std::vector<std::uint64_t> keys(2, 0);
  cc.rebalance(vectors, keys);

  const std::size_t j = static_cast<std::size_t>(keys[0]);
  std::vector<float> c(cc.centroid(j).begin(), cc.centroid(j).end());
  (void)cc.assign_and_update(c);
  EXPECT_NEAR(dot(cc.centroid(j), c), 1.0f, 1e-5f);
}

TEST(ClusterBucketsTest, ProbeOrdersByCentroidIp) {
  const std::size_t dim = 4;
  index::ClusterCentroids cc(4, dim, 11);
  // Put one vector in each of two buckets by assigning distinct axes.
  std::vector<std::uint64_t> keys;
  for (std::size_t i = 0; i < 4; ++i) {
    auto v = unit_axis(dim, i);
    keys.push_back(cc.assign_and_update(v));
  }

  std::vector<std::uint64_t> sorted = keys;
  std::sort(sorted.begin(), sorted.end());
  auto idx = index::BucketIndex::build(sorted, std::move(cc));

  auto q = unit_axis(dim, 0);
  std::size_t candidates = 0;
  const auto ranges = idx.probe(q, 2, &candidates);
  EXPECT_FALSE(ranges.empty());
  EXPECT_LE(ranges.size(), 2u);
  EXPECT_GT(candidates, 0u);

  EXPECT_THROW(index::validate_probe_radius(0), Error);
  EXPECT_THROW(index::validate_probe_radius(index::kMaxProbeCells + 1), Error);
}

TEST(ClusterBucketsTest, CsrRangesContiguous) {
  index::ClusterCentroids cc(3, 4, 1);
  std::vector<std::uint64_t> keys = {0, 0, 1, 2, 2};
  // Need matching centroid dim; build does not require counts.
  auto idx = index::BucketIndex::build(keys, std::move(cc));
  EXPECT_EQ(idx.num_cells(), 3u);
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
  index::ClusterCentroids cc(4, dim, 99);
  std::vector<std::uint64_t> keys = {2, 0, 2, 1, 0};
  store.finalize_buckets(keys, std::move(cc));
  EXPECT_TRUE(store.has_buckets());
  EXPECT_EQ(store.buckets().num_cells(), 3u);
  EXPECT_EQ(store.id_at(0), 1u);  // key 0 first after argsort
}

TEST(ClusterBucketsTest, SelfHitWithNprobe) {
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
  bp.num_buckets = 8;
  bp.rebalance_every = 0;
  bp.bucket_seed = 7;
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42, 1, bp);
  ASSERT_EQ(engine.ingest(reader).vectors_ingested, n);

  auto qe = query::QueryEngine::with_rotation(engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 5;
  params.probe_radius = 4;
  const auto hits = qe.search(vectors[0], params);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits.front().id, 0u);
}

TEST(ClusterBucketsTest, EmptyListsSkippedInProbe) {
  const std::size_t dim = 8;
  index::ClusterCentroids cc(8, dim, 3);
  // Only populate one bucket.
  auto v = unit_axis(dim, 0);
  const auto key = cc.assign_and_update(v);
  std::vector<std::uint64_t> keys = {key};
  auto idx = index::BucketIndex::build(keys, std::move(cc));

  std::size_t candidates = 0;
  const auto ranges = idx.probe(v, 8, &candidates);
  EXPECT_EQ(ranges.size(), 1u);
  EXPECT_EQ(candidates, 1u);
}
