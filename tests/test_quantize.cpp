#include <gtest/gtest.h>

#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::quantize;

TEST(QuantizeTest, Single4dBlockPositiveProjection) {
  std::vector<std::uint64_t> words(1, 0);
  const std::vector<float> v = {1.0f, 1.0f, 1.0f, 1.0f};
  EXPECT_EQ(quantize_4d_to_1bit_into(v, words), 1u);
  EXPECT_EQ(words[0], 1u);
}

TEST(QuantizeTest, Single4dBlockNegativeProjection) {
  std::vector<std::uint64_t> words(1, 0);
  const std::vector<float> v = {-1.0f, -1.0f, -1.0f, -1.0f};
  EXPECT_EQ(quantize_4d_to_1bit_into(v, words), 1u);
  EXPECT_EQ(words[0], 0u);
}

TEST(QuantizeTest, CyclicBasisChangesBit) {
  std::vector<std::uint64_t> w0(1, 0);
  const std::vector<float> v0 = {1.0f, 2.0f, 0.0f, 0.0f};
  quantize_4d_to_1bit_into(v0, w0);
  std::vector<float> v8(8, 0.0f);
  v8[4] = 1.0f;
  v8[5] = 2.0f;
  std::vector<std::uint64_t> w1(1, 0);
  quantize_4d_to_1bit_into(v8, w1);
  EXPECT_EQ(w0[0] & 1u, 1u);
  EXPECT_EQ(w1[0] & 2u, 0u);
}

TEST(QuantizeTest, PaddingPartialLastBlock) {
  std::vector<std::uint64_t> words(1, 0);
  const std::vector<float> v = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  EXPECT_EQ(quantize_4d_to_1bit_into(v, words), 2u);
  EXPECT_EQ(words[0], 0b11u);
}

TEST(QuantizeTest, BenchmarkDims) {
  EXPECT_EQ(l1_words_per_vector(256), 1u);
  std::vector<std::uint64_t> w(1, 0);
  EXPECT_EQ(quantize_4d_to_1bit_into(std::vector<float>(256, 0.0f), w), 64u);

  EXPECT_EQ(l1_words_per_vector(2048), 8u);
  std::vector<std::uint64_t> w8(8, 0);
  EXPECT_EQ(quantize_4d_to_1bit_into(std::vector<float>(2048, 0.0f), w8), 512u);

  EXPECT_EQ(l1_words_per_vector(4096), 16u);
  std::vector<std::uint64_t> w16(16, 0);
  EXPECT_EQ(quantize_4d_to_1bit_into(std::vector<float>(4096, 0.0f), w16), 1024u);
}

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

TEST(QuantizeTest, IntoMatchesAllocating) {
  std::vector<float> v(16);
  for (std::size_t i = 0; i < 16; ++i) v[i] = static_cast<float>(i) * 0.1f - 0.5f;
  const auto [alloc, n1] = quantize_4d_to_1bit(v);
  std::vector<std::uint64_t> into(alloc.size(), 0);
  const std::size_t n2 = quantize_4d_to_1bit_into(v, into);
  EXPECT_EQ(n1, n2);
  EXPECT_EQ(alloc, into);
}
