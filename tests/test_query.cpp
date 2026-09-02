#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/query/engine.hpp"
#include "vectorcache/query/query_config.hpp"
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
    if (index_ >= vectors_.size()) {
      return false;
    }
    std::copy(vectors_[index_].begin(), vectors_[index_].end(), out.begin());
    ++index_;
    return true;
  }

 private:
  std::vector<std::vector<float>> vectors_;
  std::size_t dim_;
  std::size_t index_ = 0;
};

class CapturingHook : public ingest::VectorHook {
 public:
  std::vector<float> last;
  void on_vector(std::uint64_t, std::span<const float> vector) override {
    last.assign(vector.begin(), vector.end());
  }
};

std::vector<std::vector<float>> make_vectors(std::size_t count, std::size_t dim) {
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < count; ++i) {
    std::vector<float> v(dim);
    for (std::size_t j = 0; j < dim; ++j) {
      v[j] = static_cast<float>(i + 1) * 0.1f + static_cast<float>(j) * 0.03f - 0.5f;
    }
    vectors.push_back(std::move(v));
  }
  return vectors;
}

}  // namespace

TEST(QueryDistanceTest, BitAgreementPerfectMatch) {
  const std::vector<std::uint64_t> bits = {0b10101010'10101010'10101010'10101010ULL};
  const float score = query::bit_agreement_score(bits, bits, 32);
  EXPECT_FLOAT_EQ(score, 1.0f);
}

TEST(QueryDistanceTest, BitAgreementOpposite) {
  const std::vector<std::uint64_t> a = {0};
  const std::vector<std::uint64_t> b = {0xFF};
  const float score = query::bit_agreement_score(a, b, 8);
  EXPECT_FLOAT_EQ(score, -1.0f);
}

#if VECTORCACHE_QUERY_DEPTH >= 3
TEST(QueryDistanceTest, DotF32Self) {
  const std::vector<float> v = {0.6f, 0.8f, 0.0f, 0.0f};
  EXPECT_NEAR(query::dot_f32(v, v), 1.0f, 1e-5f);
}
#endif

TEST(QueryEngineTest, SelfSimilarityTopScore) {
  const std::size_t dim = 4;
  const auto vectors = make_vectors(8, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 3;
  params.l1_block_threshold = -1.0f;
  params.l0_vector_threshold = -1.0f;

  const auto hits = query_engine.search(vectors[3], params);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 3u);
#if VECTORCACHE_QUERY_DEPTH >= 3
  EXPECT_NEAR(hits[0].score, 1.0f, 1e-4f);
#elif VECTORCACHE_QUERY_DEPTH == 2
  EXPECT_NEAR(hits[0].score, 1.0f, 1e-4f);
#else
  EXPECT_NEAR(hits[0].score, 1.0f, 1e-4f);
#endif
}

TEST(QueryEngineTest, BlockGateReducesScoresAtDepth1) {
#if VECTORCACHE_QUERY_DEPTH != 1
  GTEST_SKIP() << "depth-1 block gate test requires VECTORCACHE_QUERY_DEPTH=1";
#endif

  const std::size_t dim = 4;
  const auto vectors = make_vectors(16, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams open;
  open.k = 100;
  open.l1_block_threshold = -1.0f;

  query::QueryParams strict;
  strict.k = 100;
  strict.l1_block_threshold = 1.1f;

  const auto open_hits = query_engine.search(vectors[0], open);
  const auto strict_hits = query_engine.search(vectors[0], strict);
  EXPECT_LT(strict_hits.size(), open_hits.size());
}

#if VECTORCACHE_QUERY_DEPTH >= 2
TEST(QueryEngineTest, L0ThresholdFiltersAtDepth2) {
#if VECTORCACHE_QUERY_DEPTH != 2
  GTEST_SKIP() << "L0 threshold test requires VECTORCACHE_QUERY_DEPTH=2";
#endif

  const std::size_t dim = 4;
  const auto vectors = make_vectors(16, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams open;
  open.k = 100;
  open.l1_block_threshold = -1.0f;
  open.l0_vector_threshold = -1.0f;

  query::QueryParams strict;
  strict.k = 100;
  strict.l1_block_threshold = -1.0f;
  strict.l0_vector_threshold = 1.1f;

  const auto open_hits = query_engine.search(vectors[0], open);
  const auto strict_hits = query_engine.search(vectors[0], strict);
  EXPECT_LT(strict_hits.size(), open_hits.size());
}
#endif

TEST(QueryEngineTest, SearchPreparedMatchesSearch) {
  const std::size_t dim = 4;
  const auto vectors = make_vectors(4, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 2;

  const auto prepared = query_engine.prepare(vectors[0]);
  const auto from_prepared = query_engine.search_prepared(prepared, params);
  const auto direct = query_engine.search(vectors[0], params);

  ASSERT_EQ(from_prepared.size(), direct.size());
  for (std::size_t i = 0; i < direct.size(); ++i) {
    EXPECT_EQ(from_prepared[i].id, direct[i].id);
    EXPECT_NEAR(from_prepared[i].score, direct[i].score, 1e-5f);
  }
}

TEST(QueryEngineTest, QueryCodesMatchIngestion) {
  const std::size_t dim = 4;
  const std::vector<float> vector = {1.0f, -2.0f, 3.0f, -4.0f};
  MockReader reader({vector}, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;
  ingest_engine.ingest_with_hook(reader, &hook);

  const auto [expected_l1, _l1] = quantize::quantize_4d_to_1bit(hook.last);
  const auto& block = ingest_engine.store().partial_block();
  const auto stored_l1 = block.vector_l1(0);
  ASSERT_EQ(stored_l1.size(), expected_l1.size());
  for (std::size_t i = 0; i < expected_l1.size(); ++i) {
    EXPECT_EQ(stored_l1[i], expected_l1[i]);
  }

#if VECTORCACHE_QUERY_DEPTH >= 2
  const auto [expected_l0, _l0] = quantize::quantize_1dim_to_1bit(hook.last);
  const auto stored_l0 = block.vector_l0(0);
  ASSERT_EQ(stored_l0.size(), expected_l0.size());
  EXPECT_EQ(stored_l0[0], expected_l0[0]);
#endif
}
