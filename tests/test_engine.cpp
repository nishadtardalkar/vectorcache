#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/fwht.hpp"

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
  const std::size_t dim = 8;
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
  EXPECT_GT(report.unique_parents, 0u);
  EXPECT_EQ(engine.store().total_vectors(), n);
}

TEST(EngineTest, IngestWithHookCountsVectors) {
  MockReader reader({{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}}, 8);
  auto engine = ingest::IngestionEngine::with_rotation(8, 42);
  CountingHook hook;
  const auto report = engine.ingest_with_hook(reader, &hook);
  EXPECT_EQ(hook.count, 1u);
  EXPECT_EQ(report.vectors_ingested, 1u);
}

TEST(EngineTest, IngestStoresParentAndL0) {
  const std::size_t dim = 8;
  MockReader reader({{1.0f, -2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 1.5f, -1.5f}}, dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  const std::uint8_t expected_parent = quantize::quantize_parent_8bit(hook.last);
  const auto [expected_l0, _] = quantize::quantize_1dim_to_1bit(hook.last);

  EXPECT_EQ(engine.store().unique_parent_count(), 1u);
  EXPECT_EQ(engine.store().unique_keys()[0], expected_parent);

  const auto* group = engine.store().group_for_key(expected_parent);
  ASSERT_NE(group, nullptr);
  ASSERT_EQ(group->size(), 1u);
  EXPECT_EQ(group->id_at(0), 0u);
  ASSERT_EQ(group->vector_l0(0).size(), expected_l0.size());
  EXPECT_EQ(group->vector_l0(0)[0], expected_l0[0]);
}

TEST(EngineTest, IngestWithRotationNormalizesBeforeSrht) {
  MockReader reader({{3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}}, 8);
  auto engine = ingest::IngestionEngine::with_rotation(8, 42);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  double norm = 0.0;
  for (float x : hook.last) norm += static_cast<double>(x) * x;
  norm = std::sqrt(norm);
  EXPECT_NEAR(norm, 1.0, 1e-4);
}

TEST(EngineTest, FromRotatedQuantizesWithoutSrht) {
  const std::vector<float> vector = {0.6f, 0.8f, -0.1f, 0.2f, 0.3f, -0.4f, 0.5f, -0.6f};
  const std::uint8_t expected_parent = quantize::quantize_parent_8bit(vector);
  const auto [expected_l0, _] = quantize::quantize_1dim_to_1bit(vector);

  MockReader reader({vector}, 8);
  auto engine = ingest::IngestionEngine::from_rotated(8);
  engine.ingest(reader);

  EXPECT_EQ(engine.store().unique_keys()[0], expected_parent);
  const auto* group = engine.store().group_for_key(expected_parent);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->vector_l0(0)[0], expected_l0[0]);
}
