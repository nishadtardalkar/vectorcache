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

  EXPECT_FLOAT_EQ(codebook.centroid_at(unpack_code(words, 0, 2)), codebook.centroid_at(3));
}

TEST(QuantizeTest, BetaCodebookSymmetricAndOrdered) {
  constexpr std::size_t dim = 200;
  LloydMaxCodebook codebook(dim, 4);
  ASSERT_EQ(codebook.num_centroids(), 16u);
  ASSERT_EQ(codebook.boundaries().size(), 15u);

  // Centroids and boundaries ascending; roughly symmetric around 0.
  for (std::size_t i = 0; i + 1 < codebook.num_centroids(); ++i) {
    EXPECT_LT(codebook.centroid_at(static_cast<std::uint32_t>(i)),
              codebook.centroid_at(static_cast<std::uint32_t>(i + 1)));
  }
  for (std::size_t i = 0; i < codebook.boundaries().size(); ++i) {
    const float expect =
        0.5f * (codebook.centroid_at(static_cast<std::uint32_t>(i)) +
                codebook.centroid_at(static_cast<std::uint32_t>(i + 1)));
    EXPECT_FLOAT_EQ(codebook.boundaries()[i], expect);
  }
  const float c0 = codebook.centroid_at(0);
  const float cN = codebook.centroid_at(15);
  EXPECT_NEAR(c0, -cN, 1e-4f);
  EXPECT_GT(std::abs(c0), 0.0f);
  EXPECT_LE(std::abs(c0), 1.0f);
}

TEST(QuantizeTest, IpScaleAlphaUnitSelf) {
  constexpr std::size_t dim = 32;
  LloydMaxCodebook codebook(dim, 2);
  std::vector<float> u(dim, 0.0f);
  u[0] = 1.0f;  // already unit
  const auto [words, _] = quantize_1dim_to_nbit(u, codebook);
  const float alpha = ip_scale_alpha(u, words, codebook);
  EXPECT_GT(alpha, 1.0f);  // reconstruction shorter than unit → α > 1
}

TEST(QuantizeTest, RejectInvalidBits) {
  EXPECT_THROW(validate_bits_per_dim(0), vectorcache::Error);
  EXPECT_THROW(validate_bits_per_dim(9), vectorcache::Error);
  EXPECT_THROW(LloydMaxCodebook(64, 0), vectorcache::Error);
  EXPECT_THROW(LloydMaxCodebook(1, 2), vectorcache::Error);
}
