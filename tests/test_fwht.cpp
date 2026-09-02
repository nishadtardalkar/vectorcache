#include <cmath>

#include <gtest/gtest.h>

#include "vectorcache/error.hpp"
#include "vectorcache/transform/fwht.hpp"

using namespace vectorcache::transform;

TEST(FwhtTest, PaddedDimValues) {
  EXPECT_EQ(padded_dim(200), 256u);
  EXPECT_EQ(padded_dim(1536), 2048u);
  EXPECT_EQ(padded_dim(3072), 4096u);
  EXPECT_EQ(padded_dim(1), 1u);
  EXPECT_EQ(padded_dim(4), 4u);
}

TEST(FwhtTest, OrthonormalSingleImpulse) {
  std::vector<float> buf = {1.0f, 0.0f, 0.0f, 0.0f};
  fwht_orthonormal_in_place(buf);
  for (float v : buf) {
    EXPECT_NEAR(v, 0.5f, 1e-6f);
  }
}

TEST(FwhtTest, OrthonormalPreservesNorm) {
  std::vector<float> buf = {3.0f, -1.0f, 2.0f, 4.0f};
  float norm_before = 0.0f;
  for (float x : buf) norm_before += x * x;
  norm_before = std::sqrt(norm_before);
  fwht_orthonormal_in_place(buf);
  float norm_after = 0.0f;
  for (float x : buf) norm_after += x * x;
  norm_after = std::sqrt(norm_after);
  EXPECT_NEAR(norm_before, norm_after, 1e-5f);
}

TEST(FwhtTest, RejectsNonPowerOfTwo) {
  std::vector<float> buf = {1.0f, 2.0f, 3.0f};
  EXPECT_THROW(fwht_in_place(buf), vectorcache::Error);
}

TEST(FwhtTest, ApplySignsI8Negates) {
  std::vector<float> buf = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<std::int8_t> signs = {-1, 1, -1, 1};
  apply_signs_i8(buf, signs);
  EXPECT_FLOAT_EQ(buf[0], -1.0f);
  EXPECT_FLOAT_EQ(buf[1], 2.0f);
  EXPECT_FLOAT_EQ(buf[2], -3.0f);
  EXPECT_FLOAT_EQ(buf[3], 4.0f);
}

TEST(FwhtTest, StockhamMatchesInPlace) {
  std::vector<float> reference(256);
  std::vector<float> stockham(256);
  std::vector<float> scratch(256);
  for (std::size_t i = 0; i < reference.size(); ++i) {
    reference[i] = static_cast<float>(i) * 0.01f - 1.0f;
    stockham[i] = reference[i];
  }

  const float inv_sqrt_n = 1.0f / std::sqrt(256.0f);
  fwht_orthonormal_in_place(reference, inv_sqrt_n);
  fwht_stockham_orthonormal_in_place(stockham, scratch, inv_sqrt_n);

  for (std::size_t i = 0; i < reference.size(); ++i) {
    EXPECT_NEAR(reference[i], stockham[i], 1e-5f) << "index " << i;
  }
}
