#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/quantize/quantize.hpp"

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
    if (index_ >= vectors_.size()) return false;
    std::copy(vectors_[index_].begin(), vectors_[index_].end(), out.begin());
    ++index_;
    return true;
  }

 private:
  std::vector<std::vector<float>> vectors_;
  std::size_t dim_;
  std::size_t index_ = 0;
};

class CountingHook : public ingest::VectorHook {
 public:
  std::uint64_t count = 0;
  void on_vector(std::uint64_t, std::span<const float>) override { ++count; }
};

class CapturingHook : public ingest::VectorHook {
 public:
  std::vector<float> last;
  void on_vector(std::uint64_t, std::span<const float> vector) override {
    last.assign(vector.begin(), vector.end());
  }
};

}  // namespace

TEST(EngineTest, IngestDefaultHasNoHook) {
  const std::size_t dim = 16;
  const std::size_t n = 20;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<float> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = static_cast<float>(i + j);
    }
    vectors.push_back(std::move(v));
  }

  MockReader reader(std::move(vectors), dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  const auto report = engine.ingest(reader);

  EXPECT_EQ(report.vectors_ingested, n);
  EXPECT_EQ(engine.store().size(), n);
  EXPECT_EQ(engine.bits_per_dim(), 1u);
}

TEST(EngineTest, IngestWithHookCountsVectors) {
  std::vector<float> v(16);
  for (std::size_t i = 0; i < 16; ++i) v[i] = static_cast<float>(i + 1);
  MockReader reader({v}, 16);
  auto engine = ingest::IngestionEngine::with_rotation(16, 42);
  CountingHook hook;
  const auto report = engine.ingest_with_hook(reader, &hook);
  EXPECT_EQ(hook.count, 1u);
  EXPECT_EQ(report.vectors_ingested, 1u);
}

TEST(EngineTest, IngestStoresL0) {
  const std::size_t dim = 16;
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  MockReader reader({v}, dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  const auto [expected_l0, _] = quantize::quantize_1dim_to_nbit(hook.last, engine.codebook());
  EXPECT_EQ(engine.store().size(), 1u);
  EXPECT_EQ(engine.store().id_at(0), 0u);
  ASSERT_EQ(engine.store().vector_l0(0).size(), expected_l0.size());
  EXPECT_EQ(engine.store().vector_l0(0)[0], expected_l0[0]);
}

TEST(EngineTest, IngestWithRotationUnitNormAfterSrht) {
  std::vector<float> v(16, 0.0f);
  v[0] = 3.0f;
  v[1] = 4.0f;
  MockReader reader({v}, 16);
  auto engine = ingest::IngestionEngine::with_rotation(16, 42);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  double norm = 0.0;
  for (float x : hook.last) norm += static_cast<double>(x) * x;
  norm = std::sqrt(norm);
  EXPECT_NEAR(norm, 1.0, 1e-4);
}

TEST(EngineTest, FromRotatedQuantizesWithoutSrht) {
  std::vector<float> vector(16);
  for (std::size_t i = 0; i < 16; ++i) {
    vector[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  auto engine = ingest::IngestionEngine::from_rotated(16);
  const auto [expected_l0, _] = quantize::quantize_1dim_to_nbit(vector, engine.codebook());

  MockReader reader({vector}, 16);
  engine.ingest(reader);

  EXPECT_EQ(engine.store().size(), 1u);
  EXPECT_EQ(engine.store().vector_l0(0)[0], expected_l0[0]);
}

TEST(EngineTest, MultiBitIngest) {
  const std::size_t dim = 16;
  std::vector<float> v(dim, 0.25f);
  MockReader reader({v}, dim);
  auto engine = ingest::IngestionEngine::from_rotated(dim, 2);
  engine.ingest(reader);
  EXPECT_EQ(engine.bits_per_dim(), 2u);
  EXPECT_EQ(engine.store().l0_words_per_vec(), quantize::l0_words_per_vector(dim, 2));
  EXPECT_EQ(engine.store().size(), 1u);
}

TEST(EngineTest, IngestWithoutFinalizeThenFinalizeBuckets) {
  const std::size_t dim = 16;
  const std::size_t n = 8;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<float> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = static_cast<float>(i + j + 1);
    }
    vectors.push_back(std::move(v));
  }

  MockReader reader(std::move(vectors), dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  const auto report = engine.ingest(reader, false);
  EXPECT_EQ(report.vectors_ingested, n);
  EXPECT_EQ(engine.store().size(), n);
  EXPECT_FALSE(engine.store().has_buckets());

  auto work = engine.store().clone();
  EXPECT_FALSE(work.has_buckets());
  EXPECT_EQ(work.size(), n);

  index::ClusterCentroids centroids(work.srht_dim());
  // Seed enough buckets for keys 0..3 via online assign + unit-count splits.
  {
    std::vector<float> rotated;
    std::vector<std::uint64_t> grow_keys;
    const std::size_t dim = work.srht_dim();
    std::size_t axis = 0;
    while (centroids.num_buckets() < 4) {
      std::vector<float> v(dim, 0.0f);
      v[axis % dim] = 1.0f;
      ++axis;
      grow_keys.push_back(centroids.assign_and_update(v));
      rotated.insert(rotated.end(), v.begin(), v.end());
      bool split = true;
      while (split) {
        split = false;
        for (std::size_t j = 0; j < centroids.num_buckets(); ++j) {
          if (centroids.count(j) > 1) {
            centroids.split_bucket(j, rotated, grow_keys, /*lloyd_iters=*/0,
                                   /*steal_neighbors=*/0);
            split = true;
            break;
          }
        }
      }
      if (axis > 64) {
        break;
      }
    }
    ASSERT_GE(centroids.num_buckets(), 4u);
  }
  std::vector<std::uint64_t> keys(n, 0);
  for (std::size_t i = 0; i < n; ++i) {
    keys[i] = static_cast<std::uint64_t>(i % 4);
  }
  work.finalize_buckets(keys, std::move(centroids));
  EXPECT_TRUE(work.has_buckets());
  EXPECT_GE(work.buckets().num_cells(), 1u);
  EXPECT_EQ(work.size(), n);
}
