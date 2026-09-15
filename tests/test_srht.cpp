#include <algorithm>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>

#include "vectorcache/transform/srht.hpp"

using namespace vectorcache::transform;

namespace {

float l2_norm(const std::vector<float>& v) {
  float sum = 0.0f;
  for (float x : v) sum += x * x;
  return std::sqrt(sum);
}

float dot(const std::vector<float>& a, const std::vector<float>& b) {
  float sum = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) sum += a[i] * b[i];
  return sum;
}

}  // namespace

TEST(SrhtTest, SameSeedSameOutput) {
  SrhtRotation a(8, 42);
  SrhtRotation b(8, 42);
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> out_a(8), out_b(8);
  a.apply(input, out_a);
  b.apply(input, out_b);
  EXPECT_EQ(out_a, out_b);
}

TEST(SrhtTest, DifferentSeedDifferentOutput) {
  SrhtRotation a(8, 42);
  SrhtRotation b(8, 99);
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  std::vector<float> out_a(8), out_b(8);
  a.apply(input, out_a);
  b.apply(input, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(SrhtTest, PreservesNormNoPad) {
  SrhtRotation rot(200, 7);
  EXPECT_EQ(rot.srht_dim(), 200u);
  EXPECT_EQ(rot.block_size(), 8u);
  std::vector<float> input(200);
  for (std::size_t i = 0; i < 200; ++i) input[i] = static_cast<float>(i) * 0.01f - 1.0f;
  std::vector<float> out(rot.srht_dim());
  rot.apply(input, out);
  EXPECT_NEAR(l2_norm(input), l2_norm(out), 1e-4f);
}

TEST(SrhtTest, PreservesInnerProductNoPad) {
  SrhtRotation rot(1536, 123);
  EXPECT_EQ(rot.srht_dim(), 1536u);
  EXPECT_EQ(rot.block_size(), 512u);
  std::vector<float> x(1536), y(1536);
  for (std::size_t i = 0; i < 1536; ++i) {
    x[i] = std::sin(static_cast<float>(i));
    y[i] = std::cos(static_cast<float>(i));
  }
  std::vector<float> rx(rot.srht_dim()), ry(rot.srht_dim());
  rot.apply(x, rx);
  rot.apply(y, ry);
  EXPECT_NEAR(dot(x, y), dot(rx, ry), 1e-3f);
}

TEST(SrhtTest, ApplyInPlaceMatchesApply) {
  SrhtRotation rot(200, 42);
  std::vector<float> input(200);
  for (std::size_t i = 0; i < 200; ++i) {
    input[i] = static_cast<float>(i) * 0.01f - 1.0f;
  }

  std::vector<float> out_apply(rot.srht_dim());
  std::vector<float> out_inplace = input;
  rot.apply(input, out_apply);
  rot.apply_in_place(out_inplace);

  EXPECT_EQ(out_apply, out_inplace);
}

TEST(SrhtTest, DimEqualsOriginalForBenchmarkDatasets) {
  EXPECT_EQ(SrhtRotation(200, 0).srht_dim(), 200u);
  EXPECT_EQ(SrhtRotation(1536, 0).srht_dim(), 1536u);
  EXPECT_EQ(SrhtRotation(3072, 0).srht_dim(), 3072u);
}
