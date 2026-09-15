#include <algorithm>
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

TEST(QueryDistanceTest, AsymmetricIpSelfMatch) {
  constexpr std::size_t dim = 64;
  quantize::LloydMaxCodebook codebook(dim, 1);
  std::vector<float> q(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    q[i] = (i % 2 == 0) ? 0.1f : -0.1f;
  }
  const auto [words, _] = quantize::quantize_1dim_to_nbit(q, codebook);
  const float score = query::asymmetric_ip_score(q, words, codebook);
  // All coords contribute positively when code matches sign of q.
  EXPECT_GT(score, 0.0f);

  std::vector<float> scores(1);
  query::asymmetric_ip_batch(q, words, words.size(), 1, codebook, scores);
  EXPECT_FLOAT_EQ(scores[0], score);
}

TEST(QueryDistanceTest, AsymmetricIpPrefersMatchingCodes) {
  constexpr std::size_t dim = 32;
  quantize::LloydMaxCodebook codebook(dim, 2);
  std::vector<float> q(dim, 0.5f);
  const auto [good, _] = quantize::quantize_1dim_to_nbit(q, codebook);
  std::vector<float> opposite(dim, -0.5f);
  const auto [bad, _b] = quantize::quantize_1dim_to_nbit(opposite, codebook);
  EXPECT_GT(query::asymmetric_ip_score(q, good, codebook),
            query::asymmetric_ip_score(q, bad, codebook));
}

float reference_asymmetric_ip(std::span<const float> query,
                              std::span<const std::uint64_t> words,
                              const quantize::LloydMaxCodebook& codebook) {
  float score = 0.0f;
  for (std::size_t d = 0; d < codebook.srht_dim(); ++d) {
    const auto code = quantize::unpack_code(words, d, codebook.bits());
    score += query[d] * codebook.centroid_at(code);
  }
  return score;
}

TEST(QueryDistanceTest, LutMatchesScalarForByteAlignedBits) {
  const std::size_t dims[] = {32, 64, 256};
  const std::size_t bit_widths[] = {1, 2, 4, 8};
  for (std::size_t dim : dims) {
    for (std::size_t bits : bit_widths) {
      quantize::LloydMaxCodebook codebook(dim, bits);
      std::vector<float> q(dim);
      for (std::size_t i = 0; i < dim; ++i) {
        q[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.07f;
      }
      const auto [words, _] = quantize::quantize_1dim_to_nbit(q, codebook);
      const float ref = reference_asymmetric_ip(q, words, codebook);

      ASSERT_TRUE(query::lut_bits_supported(bits));
      query::QueryLut lut;
      query::build_query_lut(q, codebook, lut);
      ASSERT_FALSE(lut.empty()) << "dim=" << dim << " bits=" << bits;

      EXPECT_NEAR(query::asymmetric_ip_score_lut(lut, words), ref, 1e-5f)
          << "dim=" << dim << " bits=" << bits;
      EXPECT_NEAR(query::asymmetric_ip_score(q, words, codebook), ref, 1e-5f)
          << "dim=" << dim << " bits=" << bits;

      std::vector<float> batch_scores(1);
      query::asymmetric_ip_batch_lut(lut, words, words.size(), 1, codebook, q, batch_scores);
      EXPECT_NEAR(batch_scores[0], ref, 1e-5f) << "dim=" << dim << " bits=" << bits;
    }
  }
}

TEST(QueryDistanceTest, LutBatchMatchesScalarAcrossAvxWidth) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 1;
  constexpr std::size_t n = 20;  // covers 16-wide AVX path + scalar tail
  quantize::LloydMaxCodebook codebook(dim, bits);

  std::vector<float> q(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    q[i] = (i % 2 == 0) ? 0.11f : -0.09f;
  }

  const std::size_t words_per = quantize::l0_words_per_vector(dim, bits);
  std::vector<std::uint64_t> all_words(n * words_per);
  std::vector<float> refs(n);
  for (std::size_t v = 0; v < n; ++v) {
    std::vector<float> row(dim);
    for (std::size_t d = 0; d < dim; ++d) {
      row[d] = static_cast<float>(v + 1) * 0.02f + static_cast<float>(d) * 0.01f - 0.4f;
    }
    auto span = std::span<std::uint64_t>(all_words.data() + v * words_per, words_per);
    quantize::quantize_1dim_to_nbit_into(row, codebook, span);
    refs[v] = reference_asymmetric_ip(q, span, codebook);
  }

  query::QueryLut lut;
  query::build_query_lut(q, codebook, lut);
  ASSERT_FALSE(lut.empty());

  std::vector<float> scores(n);
  query::asymmetric_ip_batch_lut(lut, all_words, words_per, n, codebook, q, scores);
  for (std::size_t v = 0; v < n; ++v) {
    EXPECT_NEAR(scores[v], refs[v], 1e-5f) << "vector " << v;
  }
}

TEST(QueryDistanceTest, OddBitsUsesScalarFallback) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 3;
  quantize::LloydMaxCodebook codebook(dim, bits);
  std::vector<float> q(dim, 0.2f);
  for (std::size_t i = 0; i < dim; i += 2) {
    q[i] = -0.15f;
  }
  const auto [words, _] = quantize::quantize_1dim_to_nbit(q, codebook);
  const float ref = reference_asymmetric_ip(q, words, codebook);

  EXPECT_FALSE(query::lut_bits_supported(bits));
  query::QueryLut lut;
  query::build_query_lut(q, codebook, lut);
  EXPECT_TRUE(lut.empty());

  EXPECT_NEAR(query::asymmetric_ip_score(q, words, codebook), ref, 1e-5f);
  std::vector<float> scores(1);
  query::asymmetric_ip_batch(q, words, words.size(), 1, codebook, scores);
  EXPECT_NEAR(scores[0], ref, 1e-5f);
}

TEST(QueryEngineTest, SelfSimilarityTopHit) {
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
  const bool found_self =
      std::any_of(hits.begin(), hits.end(), [](const query::QueryHit& h) {
        return h.id == 3u;
      });
  EXPECT_TRUE(found_self);
  EXPECT_EQ(hits[0].id, 3u);
}

TEST(QueryEngineTest, FlatScanFindsBothNearVectors) {
  const std::size_t dim = 64;
  std::vector<float> a(dim, 0.01f);
  std::vector<float> b(dim, 0.01f);
  a[10] = 5.0f;
  a[20] = 4.0f;
  b[10] = 4.5f;
  b[20] = 4.2f;

  MockReader reader({a, b}, dim);
  auto ingest_engine = ingest::IngestionEngine::from_rotated(dim);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::from_rotated(ingest_engine.store());
  query::QueryParams params;
  params.k = 2;

  const auto hits = query_engine.search(a, params);
  ASSERT_EQ(hits.size(), 2u);
  const bool saw0 = std::any_of(hits.begin(), hits.end(),
                                [](const query::QueryHit& h) { return h.id == 0u; });
  const bool saw1 = std::any_of(hits.begin(), hits.end(),
                                [](const query::QueryHit& h) { return h.id == 1u; });
  EXPECT_TRUE(saw0);
  EXPECT_TRUE(saw1);
  EXPECT_EQ(hits[0].id, 0u);
}

TEST(QueryEngineTest, PreferHigherAsymmetricScore) {
  const std::size_t dim = 64;
  std::vector<float> near(dim, -1.0f);
  std::vector<float> far(dim, -1.0f);
  for (std::size_t i = 0; i < 32; ++i) {
    near[i] = 1.0f;
  }
  for (std::size_t i = 32; i < 64; ++i) {
    far[i] = 1.0f;
  }

  MockReader reader({near, far}, dim);
  auto ingest_engine = ingest::IngestionEngine::from_rotated(dim);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::from_rotated(ingest_engine.store());
  query::QueryParams params;
  params.k = 1;

  const auto hits = query_engine.search(near, params);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 0u);
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

TEST(QueryEngineTest, StoredCodesMatchIngestion) {
  const std::size_t dim = 64;
  std::vector<float> vector(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    vector[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  MockReader reader({vector}, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42);
  CapturingHook hook;
  ingest_engine.ingest_with_hook(reader, &hook);

  const auto [expected_l0, _l0] =
      quantize::quantize_1dim_to_nbit(hook.last, ingest_engine.codebook());
  EXPECT_EQ(ingest_engine.store().size(), 1u);
  EXPECT_EQ(ingest_engine.store().vector_l0(0)[0], expected_l0[0]);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  const auto prepared = query_engine.prepare(vector);
  EXPECT_EQ(prepared.rotated.size(), ingest_engine.store().srht_dim());
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

TEST(QueryEngineTest, MultiBitSearchRanksSelfFirst) {
  const std::size_t dim = 64;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < 4; ++i) {
    std::vector<float> v(dim, 0.0f);
    // Distinct sparse spikes so asymmetric IP cleanly prefers the matching code.
    v[i * 8] = 5.0f;
    v[i * 8 + 1] = 4.0f;
    vectors.push_back(std::move(v));
  }
  MockReader reader(vectors, dim);

  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42, 2);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 1;
  const auto hits = query_engine.search(vectors[2], params);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 2u);
  EXPECT_EQ(ingest_engine.store().bits_per_dim(), 2u);
}
