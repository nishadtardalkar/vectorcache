#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>

#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/hook.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/query/engine.hpp"
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
  const std::vector<std::uint64_t> bits = {0xAAAAAAAAAAAAAAAAULL};
  const float score = query::bit_agreement_score(bits, bits, 64);
  EXPECT_FLOAT_EQ(score, 1.0f);
}

TEST(QueryDistanceTest, BitAgreementOpposite) {
  const std::vector<std::uint64_t> a = {0};
  const std::vector<std::uint64_t> b = {~0ULL};
  const float score = query::bit_agreement_score(a, b, 64);
  EXPECT_FLOAT_EQ(score, -1.0f);
}

TEST(QueryDistanceTest, BitAgreementSimdFullWords) {
  // 512 bits (8 u64) exercises the AVX-512 VPOPCNT path that previously summed disagree as agree.
  std::vector<std::uint64_t> a(8, 0);
  std::vector<std::uint64_t> b(8, ~0ULL);
  EXPECT_FLOAT_EQ(query::bit_agreement_score(a, a, 512), 1.0f);
  EXPECT_FLOAT_EQ(query::bit_agreement_score(a, b, 512), -1.0f);
}

TEST(QueryDistanceTest, BitAgreementBatchMatchesScalar) {
  constexpr std::size_t words = 4;
  constexpr std::size_t bits = words * 64;
  constexpr std::size_t n = 5;
  std::vector<std::uint64_t> query(words, 0x0f0f0f0f0f0f0f0fULL);
  std::vector<std::uint64_t> data(n * words);
  for (std::size_t v = 0; v < n; ++v) {
    for (std::size_t w = 0; w < words; ++w) {
      data[v * words + w] = query[w] ^ (static_cast<std::uint64_t>(v) << w);
    }
  }
  std::vector<float> batch(n);
  query::bit_agreement_batch(query, bits, data, words, n, batch);
  for (std::size_t v = 0; v < n; ++v) {
    const float scalar = query::bit_agreement_score(
        query, std::span<const std::uint64_t>(data.data() + v * words, words), bits);
    EXPECT_NEAR(batch[v], scalar, 1e-6f);
  }

  std::vector<std::uint32_t> disagree(n);
  query::bit_agreement_batch_disagree(query, bits, data, words, n, disagree);
  for (std::size_t v = 0; v < n; ++v) {
    EXPECT_NEAR(query::score_from_disagree(disagree[v], bits), batch[v], 1e-6f);
  }
}

TEST(QueryEngineTest, SelfSimilarityTopScore) {
  const std::size_t dim = 64;
  const auto vectors = make_vectors(8, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 3;

  const auto hits = query_engine.search(vectors[3], params);
  ASSERT_FALSE(hits.empty());
  EXPECT_NEAR(hits[0].score, 1.0f, 1e-4f);
  const bool found_self =
      std::any_of(hits.begin(), hits.end(), [](const query::QueryHit& h) {
        return h.id == 3u && h.score > 0.999f;
      });
  EXPECT_TRUE(found_self);
}

TEST(QueryEngineTest, ExactGroupFindsNeighborAcrossUnstableFold) {
  // Dim 64, chunk=8. Neighbor crosses F1 (s1) on group 0 but shares F2 (x0-x1).
  // Query near F1 boundary picks F2 → exact group still contains the neighbor.
  const std::size_t dim = 64;
  const std::size_t chunk = dim / 8;

  std::vector<float> a(dim, 1.0f);
  std::vector<float> b(dim, 1.0f);
  // Group 0: keep s2 = x0-x1 identical and positive; flip s1 across zero.
  // a: [0.5, -0.4, -0.05, ...] → s1 small positive, s2=0.9
  // b: [-0.4, -0.5, -0.05, ...] → s1 negative, s2=0.1 (still > 0)
  a[0] = 0.5f;
  a[1] = -0.4f;
  b[0] = -0.4f;
  b[1] = -0.5f;
  for (std::size_t i = 2; i < chunk; ++i) {
    a[i] = -0.01f;
    b[i] = -0.01f;
  }
  // a: s1 = 0.5-0.4-0.01*6 = 0.04; s2 = 0.9
  //     2*s1²=0.0032, C*s2²=8*0.81=6.48 → pick F2, bit=1
  // b: s2 = 0.1 > 0 → same F2 bit

  MockReader reader({a, b}, dim);
  auto ingest_engine = ingest::IngestionEngine::from_rotated(dim);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::from_rotated(ingest_engine.store());
  query::QueryParams params;
  params.k = 2;

  const auto hits = query_engine.search(a, params);
  ASSERT_FALSE(hits.empty());
  const bool saw0 = std::any_of(hits.begin(), hits.end(),
                                [](const query::QueryHit& h) { return h.id == 0u; });
  const bool saw1 = std::any_of(hits.begin(), hits.end(),
                                [](const query::QueryHit& h) { return h.id == 1u; });
  EXPECT_TRUE(saw0);
  EXPECT_TRUE(saw1);

  const std::uint16_t qkey = quantize::quantize_parent_query_key(a);
  const auto* group = ingest_engine.store().group_for_key(qkey);
  ASSERT_NE(group, nullptr);
  EXPECT_GE(group->size(), 2u);
}

TEST(QueryEngineTest, SearchPreparedMatchesSearch) {
  const std::size_t dim = 64;
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
  const std::size_t dim = 64;
  std::vector<float> vector(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    vector[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  MockReader reader({vector}, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;
  ingest_engine.ingest_with_hook(reader, &hook);

  const auto expected_keys = quantize::parent_posting_keys(hook.last);
  const std::uint16_t expected_query_key = quantize::quantize_parent_query_key(hook.last);
  const auto [expected_l0, _l0] = quantize::quantize_1dim_to_1bit(hook.last);

  EXPECT_EQ(ingest_engine.store().unique_parent_count(), quantize::PARENT_POSTINGS);
  const auto* group = ingest_engine.store().group_for_key(expected_query_key);
  ASSERT_NE(group, nullptr);
  EXPECT_EQ(group->vector_l0(0)[0], expected_l0[0]);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  const auto prepared = query_engine.prepare(vector);
  EXPECT_EQ(prepared.parent_key, expected_query_key);
  EXPECT_EQ(prepared.l0[0], expected_l0[0]);
  EXPECT_TRUE(std::find(expected_keys.begin(), expected_keys.end(), prepared.parent_key) !=
              expected_keys.end());
}

TEST(QueryEngineTest, TopKKeepsHighestScores) {
  const std::size_t dim = 64;
  const auto vectors = make_vectors(16, dim);
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 5;

  const auto hits = query_engine.search(vectors[0], params);
  ASSERT_LE(hits.size(), params.k);
  for (std::size_t i = 1; i < hits.size(); ++i) {
    EXPECT_GE(hits[i - 1].score, hits[i].score);
  }
}
