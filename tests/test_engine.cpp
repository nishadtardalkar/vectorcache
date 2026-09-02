#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "vectorcache/datasets/reader.hpp"
#include "vectorcache/ingest/block.hpp"
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
  const std::size_t dim = 4;
  const std::size_t n = ingest::BLOCK_SIZE * 2 + 5;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < n; ++i) {
    vectors.push_back({static_cast<float>(i), static_cast<float>(i) + 1.0f,
                       static_cast<float>(i) + 2.0f, static_cast<float>(i) + 3.0f});
  }

  MockReader reader(std::move(vectors), dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  const auto report = engine.ingest(reader);

  EXPECT_EQ(report.vectors_ingested, n);
  EXPECT_EQ(report.full_blocks, 2u);
  EXPECT_EQ(report.partial_len, 5u);
  EXPECT_EQ(engine.store().total_vectors(), n);
}

TEST(EngineTest, IngestWithHookCountsVectors) {
  MockReader reader({{1.0f, 2.0f, 3.0f, 4.0f}}, 4);
  auto engine = ingest::IngestionEngine::with_rotation(4, 42);
  CountingHook hook;
  const auto report = engine.ingest_with_hook(reader, &hook);
  EXPECT_EQ(hook.count, 1u);
  EXPECT_EQ(report.vectors_ingested, 1u);
}

TEST(EngineTest, IngestWithRotationStoresL1Codes) {
  const std::size_t dim = 4;
  const std::size_t n = 3;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < n; ++i) {
    vectors.push_back({static_cast<float>(i), static_cast<float>(i) + 1.0f,
                       static_cast<float>(i) + 2.0f, static_cast<float>(i) + 3.0f});
  }

  MockReader reader(std::move(vectors), dim);
  auto engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;

  const std::size_t padded = transform::padded_dim(dim);
  const std::size_t words_per_vec = quantize::l1_words_per_vector(padded);
  EXPECT_EQ(engine.store().l1_words_per_vec(), words_per_vec);

  const auto report = engine.ingest_with_hook(reader, &hook);
  EXPECT_EQ(report.vectors_ingested, n);
  EXPECT_EQ(engine.store().total_vectors(), n);

  const auto [expected_codes, _] = quantize::quantize_4d_to_1bit(hook.last);
  const auto slice = engine.store().partial_block().as_slice();
  const std::size_t offset = (n - 1) * words_per_vec;
  for (std::size_t i = 0; i < words_per_vec; ++i) {
    EXPECT_EQ(slice[offset + i], expected_codes[i]);
  }
}

TEST(EngineTest, IngestWithRotationNormalizesBeforeSrht) {
  MockReader reader({{3.0f, 4.0f, 0.0f, 0.0f}}, 4);
  auto engine = ingest::IngestionEngine::with_rotation(4, 42);
  CapturingHook hook;
  engine.ingest_with_hook(reader, &hook);

  double norm = 0.0;
  for (float x : hook.last) norm += static_cast<double>(x) * x;
  norm = std::sqrt(norm);
  EXPECT_NEAR(norm, 1.0, 1e-4);
}

TEST(EngineTest, FromRotatedQuantizesWithoutSrht) {
  const std::vector<float> vector = {0.6f, 0.8f, 0.0f, 0.0f};
  const auto [expected, _] = quantize::quantize_4d_to_1bit(vector);

  MockReader reader({vector}, 4);
  auto engine = ingest::IngestionEngine::from_rotated(4);
  engine.ingest(reader);

  const auto slice = engine.store().partial_block().as_slice();
  ASSERT_EQ(slice.size(), expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(slice[i], expected[i]);
  }
}
