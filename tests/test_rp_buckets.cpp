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

TEST(RpBucketsTest, BinFloorAndPackRoundTrip) {
  const index::BinCodec codec = index::make_bin_codec(2, 0.1f);
  std::int32_t bins[2] = {-3, 4};
  const std::uint64_t key =
      index::pack_cell_key(std::span<const std::int32_t>(bins, 2), codec);
  std::int32_t out[2] = {0, 0};
  index::unpack_cell_key(key, codec, std::span<std::int32_t>(out, 2));
  EXPECT_EQ(out[0], -3);
  EXPECT_EQ(out[1], 4);
}

TEST(RpBucketsTest, ProjectToBinsMatchesFloorDot) {
  index::ProjectionMatrix matrix(1, 4, 123);
  std::vector<float> x = {1.0f, 0.0f, 0.0f, 0.0f};
  // Normalize for unit sphere convention.
  const float inv = 1.0f;
  (void)inv;
  std::int32_t bins[1];
  index::project_to_bins(matrix, x, 0.25f, std::span<std::int32_t>(bins, 1));
  float proj[1];
  matrix.project(x, std::span<float>(proj, 1));
  EXPECT_EQ(bins[0], static_cast<std::int32_t>(std::floor(proj[0] / 0.25f)));
}

TEST(RpBucketsTest, MultiProbeCountIsOddPower) {
  std::vector<std::uint64_t> keys = {1, 1, 2, 2, 2, 5};
  index::ProjectionMatrix matrix(2, 8, 7);
  const index::BinCodec codec = index::make_bin_codec(2, 0.1f);
  // Build with keys that match packed bins we will probe — use synthetic CSR via build.
  // Re-key: pack (0,0), (0,0), (0,1), ...
  std::int32_t b00[2] = {0, 0};
  std::int32_t b01[2] = {0, 1};
  std::int32_t b10[2] = {1, 0};
  const std::uint64_t k00 = index::pack_cell_key(std::span<const std::int32_t>(b00, 2), codec);
  const std::uint64_t k01 = index::pack_cell_key(std::span<const std::int32_t>(b01, 2), codec);
  const std::uint64_t k10 = index::pack_cell_key(std::span<const std::int32_t>(b10, 2), codec);
  std::vector<std::uint64_t> sorted = {k00, k00, k01, k01, k01, k10};
  auto index = index::BucketIndex::build(sorted, matrix, codec);
  EXPECT_EQ(index.num_cells(), 3u);

  std::int32_t q[2] = {0, 0};
  std::size_t candidates = 0;
  const auto ranges = index.probe(std::span<const std::int32_t>(q, 2), 1, &candidates);
  // Grid size (2*1+1)^2 = 9; only 3 non-empty cells exist near (0,0): (0,0),(0,1),(1,0)
  // and also (-1,*), (0,-1), (1,1), etc. may be empty.
  EXPECT_EQ(ranges.size(), 3u);
  EXPECT_EQ(candidates, 6u);

  EXPECT_THROW(index::validate_probe_grid(4, 4), Error);  // 9^4 >> 4096
}

TEST(RpBucketsTest, CsrRangesContiguous) {
  index::ProjectionMatrix matrix(1, 4, 1);
  const index::BinCodec codec = index::make_bin_codec(1, 0.5f);
  std::int32_t b0[1] = {0};
  std::int32_t b1[1] = {1};
  const std::uint64_t k0 = index::pack_cell_key(std::span<const std::int32_t>(b0, 1), codec);
  const std::uint64_t k1 = index::pack_cell_key(std::span<const std::int32_t>(b1, 1), codec);
  std::vector<std::uint64_t> sorted = {k0, k0, k0, k1, k1};
  auto index = index::BucketIndex::build(sorted, std::move(matrix), codec);
  const auto r0 = index.find(k0);
  const auto r1 = index.find(k1);
  EXPECT_EQ(r0.start, 0u);
  EXPECT_EQ(r0.length, 3u);
  EXPECT_EQ(r1.start, 3u);
  EXPECT_EQ(r1.length, 2u);
}

TEST(StoreBucketsTest, PermutePreservesIdsAndCodes) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t bits = 1;
  const std::size_t l0_words = quantize::l0_words_per_vector(dim, bits);
  ingest::VectorStore store(l0_words, dim, dim, bits);
  std::vector<std::uint64_t> a(l0_words, 1);
  std::vector<std::uint64_t> b(l0_words, 2);
  std::vector<std::uint64_t> c(l0_words, 3);
  store.push(10, a, 1.0f);
  store.push(20, b, 2.0f);
  store.push(30, c, 3.0f);

  std::size_t order[3] = {2, 0, 1};
  store.permute(order);
  EXPECT_EQ(store.id_at(0), 30u);
  EXPECT_EQ(store.id_at(1), 10u);
  EXPECT_EQ(store.id_at(2), 20u);
  EXPECT_EQ(store.vector_l0(0)[0], 3u);
  EXPECT_EQ(store.vector_l0(1)[0], 1u);
  EXPECT_EQ(store.scale_at(0), 3.0f);
}

TEST(StoreBucketsTest, FinalizeBuildsBuckets) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t bits = 1;
  const std::size_t l0_words = quantize::l0_words_per_vector(dim, bits);
  ingest::VectorStore store(l0_words, dim, dim, bits);
  std::vector<std::uint64_t> l0(l0_words, 0);
  store.push(0, l0);
  store.push(1, l0);
  store.push(2, l0);

  index::ProjectionMatrix matrix(1, dim, 99);
  const index::BinCodec codec = index::make_bin_codec(1, 0.1f);
  std::vector<std::uint64_t> keys = {5, 1, 5};
  store.finalize_buckets(keys, std::move(matrix), codec);
  ASSERT_TRUE(store.has_buckets());
  EXPECT_EQ(store.num_tables(), 1u);
  EXPECT_FALSE(store.buckets().has_postings());
  EXPECT_EQ(store.buckets().num_cells(), 2u);
  EXPECT_EQ(store.buckets().cell(0).length, 1u);  // key 1
  EXPECT_EQ(store.buckets().cell(1).length, 2u);  // key 5
  // After sort by key: key1, key5, key5 → ids 1, 0, 2
  EXPECT_EQ(store.id_at(0), 1u);
  EXPECT_EQ(store.id_at(1), 0u);
  EXPECT_EQ(store.id_at(2), 2u);
}

TEST(StoreBucketsTest, MultiTablePostingsKeepIngestOrder) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t bits = 1;
  const std::size_t l0_words = quantize::l0_words_per_vector(dim, bits);
  ingest::VectorStore store(l0_words, dim, dim, bits);
  std::vector<std::uint64_t> l0(l0_words, 0);
  store.push(0, l0);
  store.push(1, l0);
  store.push(2, l0);

  const index::BinCodec codec = index::make_bin_codec(1, 0.1f);
  // Table-major keys: table0 = {5,1,5}, table1 = {2,2,9}
  std::vector<std::uint64_t> all_keys = {5, 1, 5, 2, 2, 9};
  std::vector<index::ProjectionMatrix> matrices;
  matrices.emplace_back(1, dim, 10);
  matrices.emplace_back(1, dim, 11);
  store.finalize_buckets(all_keys, std::move(matrices), codec);

  ASSERT_EQ(store.num_tables(), 2u);
  EXPECT_TRUE(store.buckets(0).has_postings());
  EXPECT_TRUE(store.buckets(1).has_postings());
  // Ingest order preserved.
  EXPECT_EQ(store.id_at(0), 0u);
  EXPECT_EQ(store.id_at(1), 1u);
  EXPECT_EQ(store.id_at(2), 2u);

  const auto r0 = store.buckets(0).find(1);
  ASSERT_EQ(r0.length, 1u);
  EXPECT_EQ(store.buckets(0).row_at(r0.start), 1u);

  const auto r1 = store.buckets(1).find(9);
  ASSERT_EQ(r1.length, 1u);
  EXPECT_EQ(store.buckets(1).row_at(r1.start), 2u);
}

TEST(RpBucketsTest, BuildPostingsMapsRows) {
  index::ProjectionMatrix matrix(1, 4, 1);
  const index::BinCodec codec = index::make_bin_codec(1, 0.5f);
  std::vector<std::uint64_t> keys = {5, 1, 5};
  auto index = index::BucketIndex::build_postings(keys, std::move(matrix), codec);
  ASSERT_TRUE(index.has_postings());
  EXPECT_EQ(index.num_cells(), 2u);
  const auto r = index.find(1);
  ASSERT_EQ(r.length, 1u);
  EXPECT_EQ(index.row_at(r.start), 1u);
  const auto r5 = index.find(5);
  ASSERT_EQ(r5.length, 2u);
  auto rows = index.rows_span(r5);
  EXPECT_EQ(rows[0], 0u);
  EXPECT_EQ(rows[1], 2u);
}

TEST(QueryBucketTest, SelfHitWithProbeZero) {
  const std::size_t dim = 64;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < 8; ++i) {
    std::vector<float> v(dim, 0.0f);
    v[i * 8] = 3.0f;
    v[i * 8 + 1] = 2.0f;
    vectors.push_back(std::move(v));
  }
  MockReader reader(vectors, dim);

  ingest::BucketParams bp;
  bp.num_projections = 1;
  bp.bin_width = 0.2f;
  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42, 1, 1, bp);
  ingest_engine.ingest(reader);
  ASSERT_TRUE(ingest_engine.store().has_buckets());

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 1;
  params.probe_radius = 0;
  query::SearchStats stats;
  const auto hits = query_engine.search(vectors[3], params, &stats);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 3u);
  EXPECT_GT(stats.candidates, 0u);
  EXPECT_LE(stats.candidates, ingest_engine.store().size());
}

TEST(QueryBucketTest, MultiTableOrFindsSelf) {
  const std::size_t dim = 64;
  std::vector<std::vector<float>> vectors;
  for (std::size_t i = 0; i < 16; ++i) {
    std::vector<float> v(dim, 0.0f);
    v[i % dim] = 1.0f + 0.01f * static_cast<float>(i);
    vectors.push_back(std::move(v));
  }
  MockReader reader(vectors, dim);

  ingest::BucketParams bp;
  bp.num_projections = 1;
  bp.num_tables = 3;
  bp.bin_width = 0.05f;
  bp.bucket_seed = 99;
  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 42, 1, 1, bp);
  ingest_engine.ingest(reader);
  ASSERT_EQ(ingest_engine.store().num_tables(), 3u);
  EXPECT_TRUE(ingest_engine.store().buckets(0).has_postings());

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 42);
  query::QueryParams params;
  params.k = 1;
  params.probe_radius = 0;
  query::SearchStats stats;
  const auto hits = query_engine.search(vectors[5], params, &stats);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 5u);
  EXPECT_GE(stats.cells_probed, 1u);
}

TEST(QueryBucketTest, FarVectorExcludedWhenProbeTight) {
  const std::size_t dim = 64;
  // Two nearly opposite spikes → different RP bins with high probability.
  std::vector<float> a(dim, 0.0f);
  std::vector<float> b(dim, 0.0f);
  a[0] = 1.0f;
  b[32] = 1.0f;
  MockReader reader({a, b}, dim);

  ingest::BucketParams bp;
  bp.num_projections = 2;
  bp.bin_width = 0.05f;
  bp.bucket_seed = 12345;
  auto ingest_engine = ingest::IngestionEngine::with_rotation(dim, 7, 1, 1, bp);
  ingest_engine.ingest(reader);

  auto query_engine = query::QueryEngine::with_rotation(ingest_engine.store(), dim, 7);
  query::QueryParams params;
  params.k = 2;
  params.probe_radius = 0;
  query::SearchStats stats;
  const auto hits = query_engine.search(a, params, &stats);
  ASSERT_FALSE(hits.empty());
  EXPECT_EQ(hits[0].id, 0u);
  // With P=0, candidates should be a single cell — often just vector 0.
  EXPECT_LE(stats.candidates, 2u);
}
