#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/error.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/engine.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/query/engine.hpp"
#include "vectorcache/transform/normalize.hpp"
#include "vectorcache/transform/srht.hpp"

using namespace vectorcache;

namespace {

class MockReader : public datasets::DatasetReader {
 public:
  MockReader(std::vector<std::vector<float>> vectors, std::size_t dim)
      : vectors_(std::move(vectors)), dim_(dim) {}

  datasets::DatasetMeta meta() const override { return {dim_, vectors_.size(), "mock"}; }

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

}  // namespace

TEST(PairHashTest, FoldDeterministicAndCursorResets) {
  index::PairHash hash(4, 123, 0.25f);
  std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f, 0.5f, -0.5f, 0.25f, 0.75f};
  const float a = hash.fold(x);
  const float b = hash.fold(x);
  EXPECT_FLOAT_EQ(a, b);
}

TEST(PairHashTest, OddDimPassThrough) {
  index::PairHash hash(1, 1, 0.1f);
  std::vector<float> x = {0.0f, 0.0f, 0.42f};  // first pair → ~0, leftover 0.42
  const float s = hash.fold(x);
  EXPECT_TRUE(std::isfinite(s));
  EXPECT_GE(s, -1.0f);
  EXPECT_LE(s, 1.0f);
}

TEST(PairHashTest, FoldToBinUsesUniformMap) {
  index::PairHash hash(2, 7, 0.25f);
  std::vector<float> x = {1.0f, 0.0f, 0.0f, 1.0f};
  const float s = std::clamp(hash.fold(x), -1.0f, 1.0f);
  const float u = 0.5f * (s + 1.0f);
  const float w = 0.25f;
  const std::int32_t expected = static_cast<std::int32_t>(std::floor(u / w));
  EXPECT_EQ(index::fold_to_bin(hash, x, w), expected);
}

TEST(PairHashTest, RidgeFoldContinuousNearZeroPair) {
  // Unit-normalize would send (ε,0) and (-ε,0) to opposite hemispheres; ridge keeps them close.
  index::PairHash hash(1, 42, 1.0f);
  const float eps = 1e-4f;
  std::vector<float> pos = {eps, 0.0f};
  std::vector<float> neg = {-eps, 0.0f};
  const float s_pos = hash.fold(pos);
  const float s_neg = hash.fold(neg);
  EXPECT_NEAR(s_pos, 0.0f, 1e-3f);
  EXPECT_NEAR(s_neg, 0.0f, 1e-3f);
  EXPECT_LT(std::fabs(s_pos - s_neg), 1e-3f);
}

TEST(PairHashTest, PackBinRoundTripSigned) {
  EXPECT_EQ(static_cast<std::int32_t>(index::pack_bin(-3)), -3);
  EXPECT_EQ(static_cast<std::int32_t>(index::pack_bin(0)), 0);
  EXPECT_EQ(static_cast<std::int32_t>(index::pack_bin(4)), 4);
}

TEST(PairHashTest, OneDProbeOrderedByAbsOffset) {
  index::PairHash hash(1, 1, 0.1f);
  const float w = 0.1f;
  std::vector<std::uint64_t> sorted = {index::pack_bin(0), index::pack_bin(0), index::pack_bin(1),
                                       index::pack_bin(2)};
  auto idx = index::BucketIndex::build(sorted, std::move(hash), w);
  EXPECT_EQ(idx.num_cells(), 3u);

  std::size_t candidates = 0;
  const auto ranges = idx.probe(1, 1, &candidates);
  EXPECT_EQ(ranges.size(), 3u);
  EXPECT_EQ(candidates, 4u);

  EXPECT_THROW(index::validate_probe_radius(index::kMaxProbeCells), Error);
}

TEST(PairHashTest, CsrRangesContiguous) {
  index::PairHash hash(1, 1, 0.1f);
  const float w = 0.5f;
  std::vector<std::uint64_t> sorted = {index::pack_bin(0), index::pack_bin(0), index::pack_bin(1)};
  auto idx = index::BucketIndex::build(sorted, std::move(hash), w);
  EXPECT_EQ(idx.num_cells(), 2u);
  EXPECT_EQ(idx.cell(0).start, 0u);
  EXPECT_EQ(idx.cell(0).length, 2u);
  EXPECT_EQ(idx.cell(1).start, 2u);
  EXPECT_EQ(idx.cell(1).length, 1u);
  EXPECT_EQ(idx.find(index::pack_bin(0)).length, 2u);
  EXPECT_EQ(idx.find(index::pack_bin(2)).length, 0u);
}

TEST(PairHashTest, FinalizeBuildsBuckets) {
  const std::size_t dim = 4;
  const std::size_t bits = 1;
  const std::size_t block_dims = 1;
  const std::size_t l0 = quantize::l0_words_per_vector(dim, bits, block_dims);
  ingest::VectorStore store(l0, dim, dim, bits, block_dims);
  std::vector<std::uint64_t> code(l0, 0);
  store.push(0, code, 1.0f);
  store.push(1, code, 1.0f);
  store.push(2, code, 1.0f);

  std::vector<std::uint64_t> keys = {index::pack_bin(5), index::pack_bin(1), index::pack_bin(5)};
  index::PairHash hash(2, 99, index::default_fold_ridge(dim));
  store.finalize_buckets(keys, std::move(hash), 0.1f);
  ASSERT_TRUE(store.has_buckets());
  EXPECT_EQ(store.buckets().num_cells(), 2u);
  EXPECT_EQ(store.buckets().cell(0).length, 1u);  // key 1
  EXPECT_EQ(store.buckets().cell(1).length, 2u);  // key 5
  EXPECT_EQ(store.id_at(0), 1u);
}

TEST(PairHashTest, SelfHitWithProbeZero) {
  const std::size_t dim = 8;
  std::vector<std::vector<float>> vectors;
  for (int i = 0; i < 5; ++i) {
    std::vector<float> v(dim, 0.0f);
    v[static_cast<std::size_t>(i) % dim] = 3.0f + static_cast<float>(i);
    v[(static_cast<std::size_t>(i) + 1) % dim] = 2.0f;
    vectors.push_back(std::move(v));
  }
  MockReader reader(vectors, dim);
  ingest::BucketParams bp;
  bp.num_pair_dirs = 4;
  bp.bin_width = 0.05f;
  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42, 1, 1, bp);
  ASSERT_EQ(ingest_engine.ingest(reader).vectors_ingested, 5u);
  ASSERT_TRUE(ingest_engine.store().has_buckets());

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 5;
  params.probe_radius = 1;
  const auto hits = query_engine.search(vectors[2], params);
  ASSERT_FALSE(hits.empty());
  const bool found_self = std::any_of(hits.begin(), hits.end(), [](const query::QueryHit& h) {
    return h.id == 2u;
  });
  EXPECT_TRUE(found_self);
}

TEST(PairHashTest, FarVectorExcludedWhenProbeTight) {
  const std::size_t dim = 8;
  std::vector<float> a(dim, 0.0f);
  a[0] = 1.0f;
  std::vector<float> b(dim, 0.0f);
  b[dim - 1] = 1.0f;
  std::vector<std::vector<float>> vectors = {a, b};
  MockReader reader(vectors, dim);

  ingest::BucketParams bp;
  bp.num_pair_dirs = 8;
  bp.bin_width = 0.05f;
  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 7, 1, 1, bp);
  ASSERT_EQ(ingest_engine.ingest(reader).vectors_ingested, 2u);

  // Fold must use post-SRHT coords to match ingest/query bins.
  const auto& buckets = ingest_engine.store().buckets();
  transform::SrhtRotation rotation(dim, 7);
  std::vector<float> a_rot(rotation.srht_dim(), 0.0f);
  std::copy(a.begin(), a.end(), a_rot.begin());
  transform::l2_normalize_in_place(std::span<float>(a_rot.data(), dim));
  rotation.apply_in_place(a_rot);
  const std::int32_t bin_a =
      index::fold_to_bin(buckets.hash(), a_rot, bp.bin_width);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 7);
  const auto prepared = query_engine.prepare(a);
  EXPECT_EQ(prepared.query_bin, bin_a);

  query::QueryParams params;
  params.k = 2;
  params.probe_radius = 0;
  query::SearchStats stats;
  const auto hits = query_engine.search(a, params, &stats);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 0u);
  if (stats.candidates == 1) {
    EXPECT_EQ(hits.size(), 1u);
  }
}
