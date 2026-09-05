#include <gtest/gtest.h>

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

TEST(QuantizeTest, Parent8bitEqualChunks) {
  std::vector<float> v(8, 0.0f);
  // chunk0 sum > 0, chunk1 sum < 0, ... alternating for 8 unit chunks of size 1
  for (std::size_t i = 0; i < 8; ++i) {
    v[i] = (i % 2 == 0) ? 1.0f : -1.0f;
  }
  EXPECT_EQ(quantize_parent_8bit(v), 0b01010101u);
}

TEST(QuantizeTest, Parent8bitSignOfSum) {
  std::vector<float> v(16, 0.0f);
  // Two dims per chunk. Chunk0: 1+(-2) < 0 → bit0=0; chunk1: 3+4 > 0 → bit1=1; rest zero → 1
  v[0] = 1.0f;
  v[1] = -2.0f;
  v[2] = 3.0f;
  v[3] = 4.0f;
  const std::uint8_t key = quantize_parent_8bit(v);
  EXPECT_EQ(key & 0b11u, 0b10u);
}

TEST(QuantizeTest, Parent8bitBenchmarkDims) {
  EXPECT_EQ(quantize_parent_8bit(std::vector<float>(256, 1.0f)), 0xFFu);
  EXPECT_EQ(quantize_parent_8bit(std::vector<float>(2048, -1.0f)), 0x00u);
  EXPECT_EQ(quantize_parent_8bit(std::vector<float>(4096, 0.0f)), 0xFFu);  // sum==0 → >= 0 → 1
}

TEST(QuantizeTest, Parent8bitRejectsBadDim) {
  EXPECT_THROW(quantize_parent_8bit(std::vector<float>(7, 1.0f)), vectorcache::Error);
}
