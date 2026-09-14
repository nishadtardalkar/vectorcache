#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
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
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42, 2);
  const auto report = engine.ingest(reader);

  EXPECT_EQ(report.vectors_ingested, n);
  EXPECT_GT(report.unique_parents, 0u);
  EXPECT_EQ(engine.store().total_vectors(), n);
}

TEST(EngineTest, IngestWithHookCountsVectors) {
  std::vector<float> v(16);
  for (std::size_t i = 0; i < 16; ++i) v[i] = static_cast<float>(i + 1);
  MockReader reader({v}, 16);
  auto engine = ingest::IngestionEngine::with_rotation(16, 42, 2);
  CountingHook hook;
  const auto report = engine.ingest_with_hook(reader, &hook);
  EXPECT_EQ(hook.count, 1u);
  EXPECT_EQ(report.vectors_ingested, 1u);
}

TEST(EngineTest, IngestStoresSupportKeyAndL0) {
  const std::size_t dim = 16;
  const std::size_t top_d = 2;
  std::vector<float> v(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  MockReader reader({v}, dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42, top_d);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  // Hook sees post-SRHT buffer; L0 must match that.
  const auto [expected_l0, _] = quantize::quantize_1dim_to_1bit(hook.last);
  EXPECT_EQ(engine.store().total_vectors(), 1u);
  EXPECT_EQ(engine.store().unique_parent_count(), 1u);

  // Reconstruct support key from pre-SRHT norm of original: re-ingest path stores one key.
  const auto* any = engine.store().find(engine.store().unique_keys()[0]);
  ASSERT_NE(any, nullptr);
  ASSERT_EQ(any->size(), 1u);
  EXPECT_EQ(any->id_at(0), 0u);
  ASSERT_EQ(any->vector_l0(0).size(), expected_l0.size());
  EXPECT_EQ(any->vector_l0(0)[0], expected_l0[0]);
}

TEST(EngineTest, IngestWithRotationUnitNormAfterSrht) {
  std::vector<float> v(16, 0.0f);
  v[0] = 3.0f;
  v[1] = 4.0f;
  MockReader reader({v}, 16);
  auto engine = ingest::IngestionEngine::with_rotation(16, 42, 2);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  double norm = 0.0;
  for (float x : hook.last) norm += static_cast<double>(x) * x;
  norm = std::sqrt(norm);
  EXPECT_NEAR(norm, 1.0, 1e-4);
}

TEST(EngineTest, FromRotatedQuantizesWithoutSrht) {
  const std::size_t top_d = 2;
  std::vector<float> vector(16);
  for (std::size_t i = 0; i < 16; ++i) {
    vector[i] = static_cast<float>(i) * 0.1f - 0.5f;
  }
  const auto expected_key = quantize::quantize_support_key(vector, top_d);
  const auto [expected_l0, _] = quantize::quantize_1dim_to_1bit(vector);

  MockReader reader({vector}, 16);
  auto engine = ingest::IngestionEngine::from_rotated(16, top_d);
  engine.ingest(reader);

  EXPECT_EQ(engine.store().unique_parent_count(), 1u);
  const auto* group = engine.store().find(expected_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->vector_l0(0)[0], expected_l0[0]);
}
