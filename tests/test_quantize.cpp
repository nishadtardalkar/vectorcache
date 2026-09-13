#include <algorithm>
#include <cmath>
#include <gtest/gtest.h>
#include <set>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::quantize;

TEST(QuantizeTest, L0SignBits) {
  std::vector<std::uint64_t> words(1, 0);
  const std::vector<float> v = {1.0f, -1.0f, 0.5f, -0.5f};
  EXPECT_EQ(quantize_1dim_to_1bit_into(v, words), 4u);
  EXPECT_EQ(words[0], 0b0101u);
}

TEST(QuantizeTest, L0BenchmarkDims) {
  EXPECT_EQ(l0_words_per_vector(256), 4u);
  std::vector<std::uint64_t> w(4, 0);
  EXPECT_EQ(quantize_1dim_to_1bit_into(std::vector<float>(256, 1.0f), w), 256u);
  EXPECT_EQ(w[0], ~0ull);

  EXPECT_EQ(l0_words_per_vector(2048), 32u);
  EXPECT_EQ(l0_words_per_vector(4096), 64u);
}

TEST(QuantizeTest, L0IntoMatchesAllocating) {
  std::vector<float> v(16);
  for (std::size_t i = 0; i < 16; ++i) v[i] = static_cast<float>(i) * 0.1f - 0.5f;
  const auto [alloc, n1] = quantize_1dim_to_1bit(v);
  std::vector<std::uint64_t> into(alloc.size(), 0);
  const std::size_t n2 = quantize_1dim_to_1bit_into(v, into);
  EXPECT_EQ(n1, n2);
  EXPECT_EQ(alloc, into);
}

TEST(QuantizeTest, ParentFoldAllOnes) {
  // Every chunk all +1: s1 > 0, s2 = 0 → both bits 1 (s2 >= 0).
  const auto folds = quantize_parent_fold_bits(std::vector<float>(256, 1.0f));
  EXPECT_EQ(folds.b1, 0xFFu);
  EXPECT_EQ(folds.b2, 0xFFu);
}

TEST(QuantizeTest, ParentFoldAllNeg) {
  // All -1: s1 < 0 → b1=0; s2 = 0 → b2=1.
  const auto folds = quantize_parent_fold_bits(std::vector<float>(256, -1.0f));
  EXPECT_EQ(folds.b1, 0x00u);
  EXPECT_EQ(folds.b2, 0xFFu);
}

TEST(QuantizeTest, ParentFoldCoordDiff) {
  // Dim 16 → chunk=2. Per chunk [a,b]: s1=a+b, s2=a-b (same as before for C=2).
  std::vector<float> v(16, 0.0f);
  // Group 0: [1, -2] → s1=-1 → b1=0; s2=3 → b2=1
  v[0] = 1.0f;
  v[1] = -2.0f;
  // Group 1: [3, 4] → s1=7 → b1=1; s2=-1 → b2=0
  v[2] = 3.0f;
  v[3] = 4.0f;
  // Remaining groups zero: s1=0 → b1=1; s2=0 → b2=1
  const auto folds = quantize_parent_fold_bits(v);
  EXPECT_EQ(folds.b1 & 0b11u, 0b10u);
  EXPECT_EQ(folds.b2 & 0b11u, 0b01u);
  EXPECT_EQ((folds.b1 >> 2) & 0x3Fu, 0x3Fu);
  EXPECT_EQ((folds.b2 >> 2) & 0x3Fu, 0x3Fu);
}

TEST(QuantizeTest, ParentFoldOddChunkDim24) {
  // dim=24 → chunk=3 (odd). F2 = x0 - x1 still valid.
  std::vector<float> v(24, 0.0f);
  // Group 0: [2, -1, 0] → s1=1 → b1=1; s2=3 → b2=1
  v[0] = 2.0f;
  v[1] = -1.0f;
  v[2] = 0.0f;
  const auto folds = quantize_parent_fold_bits(v);
  EXPECT_EQ(folds.b1 & 1u, 1u);
  EXPECT_EQ(folds.b2 & 1u, 1u);
}

TEST(QuantizeTest, ParentPostingCountAndUniqueness) {
  const auto keys = parent_posting_keys(std::vector<float>(16, 1.0f));
  EXPECT_EQ(keys.size(), PARENT_POSTINGS);
  std::set<std::uint16_t> unique(keys.begin(), keys.end());
  EXPECT_EQ(unique.size(), PARENT_POSTINGS);
}

TEST(QuantizeTest, QueryPicksFartherFold) {
  // Dim 16, chunk=2. Make group0 near F1 boundary (|s1| small) and far on F2.
  // [ε, -ε]: s1≈0, s2=2ε → pick F2.
  std::vector<float> v(16, 1.0f);
  v[0] = 0.01f;
  v[1] = -0.01f;
  const std::uint16_t key = quantize_parent_query_key(v);
  // Group 0 should use fold F2 (bit1 of nibble = 1) with bit = sign(s2)=1 → nibble 0b11
  EXPECT_EQ(key & 0x3u, 0b11u);
}

TEST(QuantizeTest, QueryKeyIsOneOfPostings) {
  std::vector<float> v(64);
  for (std::size_t i = 0; i < 64; ++i) {
    v[i] = static_cast<float>(i) * 0.07f - 1.1f;
  }
  const auto keys = parent_posting_keys(v);
  const std::uint16_t qkey = quantize_parent_query_key(v);
  EXPECT_TRUE(std::find(keys.begin(), keys.end(), qkey) != keys.end());
}

TEST(QuantizeTest, ParentRejectsBadDim) {
  EXPECT_THROW(quantize_parent_fold_bits(std::vector<float>(7, 1.0f)), vectorcache::Error);
  // Chunk size 1 (dim=8) rejected — no F2 ⊥ all-ones in R^1.
  EXPECT_THROW(quantize_parent_fold_bits(std::vector<float>(8, 1.0f)), vectorcache::Error);
  EXPECT_NO_THROW(quantize_parent_fold_bits(std::vector<float>(16, 1.0f)));
  EXPECT_NO_THROW(quantize_parent_fold_bits(std::vector<float>(24, 1.0f)));
}
