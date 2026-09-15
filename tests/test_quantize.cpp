#include "vectorcache/quantize/quantize.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

using namespace vectorcache::quantize;

TEST(QuantizeTest, L0AllPositive) {
  const std::vector<float> v(64, 1.0f);
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 64u);
  EXPECT_EQ(words.size(), 1u);
  EXPECT_EQ(words[0], ~0ull);
}

TEST(QuantizeTest, L0AllNegative) {
  const std::vector<float> v(64, -1.0f);
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 64u);
  EXPECT_EQ(words[0], 0ull);
}

TEST(QuantizeTest, L0MixedSigns) {
  std::vector<float> v(8, -1.0f);
  v[0] = 1.0f;
  v[3] = 1.0f;
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 8u);
  EXPECT_EQ(words.size(), 1u);
  EXPECT_EQ(words[0] & 0xFFull, 0b00001001ull);
}

TEST(QuantizeTest, L0WordsPerVector) {
  EXPECT_EQ(l0_words_per_vector(64), 1u);
  EXPECT_EQ(l0_words_per_vector(65), 2u);
  EXPECT_EQ(l0_bits_per_vector(256), 256u);
}
