#include "vectorcache/quantize/quantize.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/error.hpp"

using namespace vectorcache::quantize;

TEST(QuantizeTest, WordsAndBitsScaleWithBitsPerDim) {
  EXPECT_EQ(l0_bits_per_vector(64, 1), 64u);
  EXPECT_EQ(l0_words_per_vector(64, 1), 1u);
  EXPECT_EQ(l0_bits_per_vector(64, 2), 128u);
  EXPECT_EQ(l0_words_per_vector(64, 2), 2u);
  EXPECT_EQ(l0_bits_per_vector(65, 1), 65u);
  EXPECT_EQ(l0_words_per_vector(65, 1), 2u);
  EXPECT_EQ(l0_bits_per_vector(256, 3), 768u);
  EXPECT_EQ(l0_words_per_vector(256, 3), 12u);
}

TEST(QuantizeTest, Bits1MatchesSignIndices) {
  constexpr std::size_t dim = 64;
  LloydMaxCodebook codebook(dim, 1);
  ASSERT_EQ(codebook.num_centroids(), 2u);
  EXPECT_LT(codebook.centroid_at(0), 0.0f);
  EXPECT_GT(codebook.centroid_at(1), 0.0f);

  std::vector<float> v(dim, -1.0f);
  v[0] = 1.0f;
  v[3] = 1.0f;
  const auto [words, bits] = quantize_1dim_to_nbit(v, codebook);
  EXPECT_EQ(bits, dim);
  EXPECT_EQ(words.size(), 1u);
  // Positive dims encode to index 1; negatives to 0.
  EXPECT_EQ(words[0] & 0xFull, 0b1001ull);
  EXPECT_EQ(unpack_code(words, 0, 1), 1u);
  EXPECT_EQ(unpack_code(words, 1, 1), 0u);
  EXPECT_EQ(unpack_code(words, 3, 1), 1u);
}

TEST(QuantizeTest, Bits2NearestCentroidAndPack) {
  constexpr std::size_t dim = 8;
  LloydMaxCodebook codebook(dim, 2);
  ASSERT_EQ(codebook.num_centroids(), 4u);

  std::vector<float> v(dim, 0.0f);
  // Far positive / far negative should map to extreme centroids.
  v[0] = 10.0f;
  v[1] = -10.0f;
  const auto [words, bits] = quantize_1dim_to_nbit(v, codebook);
  EXPECT_EQ(bits, dim * 2);
  EXPECT_EQ(unpack_code(words, 0, 2), 3u);
  EXPECT_EQ(unpack_code(words, 1, 2), 0u);

  // Round-trip: reconstructed value is the chosen centroid.
  EXPECT_FLOAT_EQ(codebook.centroid_at(unpack_code(words, 0, 2)), codebook.centroid_at(3));
}

TEST(QuantizeTest, Bits1CentroidsMatchAnalytical) {
  constexpr std::size_t dim = 256;
  LloydMaxCodebook codebook(dim, 1);
  const float expected = std::sqrt(2.0f / (static_cast<float>(3.14159265f) * static_cast<float>(dim)));
  EXPECT_NEAR(codebook.centroid_at(1), expected, 1e-4f);
  EXPECT_NEAR(codebook.centroid_at(0), -expected, 1e-4f);
}

TEST(QuantizeTest, RejectInvalidBits) {
  EXPECT_THROW(validate_bits_per_dim(0), vectorcache::Error);
  EXPECT_THROW(validate_bits_per_dim(9), vectorcache::Error);
  EXPECT_THROW(LloydMaxCodebook(64, 0), vectorcache::Error);
}
