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

std::vector<float> pad_vector(const std::vector<float>& v, std::size_t n) {
  std::vector<float> out(n, 0.0f);
  std::copy(v.begin(), v.end(), out.begin());
  return out;
}

}  // namespace

TEST(SrhtTest, SameSeedSameOutput) {
  SrhtRotation a(4, 42);
  SrhtRotation b(4, 42);
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out_a(4), out_b(4);
  a.apply(input, out_a);
  b.apply(input, out_b);
  EXPECT_EQ(out_a, out_b);
}

TEST(SrhtTest, DifferentSeedDifferentOutput) {
  SrhtRotation a(4, 42);
  SrhtRotation b(4, 99);
  std::vector<float> input = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> out_a(4), out_b(4);
  a.apply(input, out_a);
  b.apply(input, out_b);
  EXPECT_NE(out_a, out_b);
}

TEST(SrhtTest, PreservesNormWithPadding) {
  SrhtRotation rot(200, 7);
  EXPECT_EQ(rot.padded_dim(), 256u);
  std::vector<float> input(200);
  for (std::size_t i = 0; i < 200; ++i) input[i] = static_cast<float>(i) * 0.01f - 1.0f;
  std::vector<float> out(rot.padded_dim());
  rot.apply(input, out);
  EXPECT_NEAR(l2_norm(input), l2_norm(out), 1e-4f);
}

TEST(SrhtTest, PreservesInnerProductWithPadding) {
  SrhtRotation rot(1536, 123);
  EXPECT_EQ(rot.padded_dim(), 2048u);
  std::vector<float> x(1536), y(1536);
  for (std::size_t i = 0; i < 1536; ++i) {
    x[i] = std::sin(static_cast<float>(i));
    y[i] = std::cos(static_cast<float>(i));
  }
  std::vector<float> rx(rot.padded_dim()), ry(rot.padded_dim());
  rot.apply(x, rx);
  rot.apply(y, ry);
  const auto x_pad = pad_vector(x, rot.padded_dim());
  const auto y_pad = pad_vector(y, rot.padded_dim());
  EXPECT_NEAR(dot(x_pad, y_pad), dot(rx, ry), 1e-3f);
}

TEST(SrhtTest, ApplyInPlaceMatchesApply) {
  SrhtRotation rot(200, 42);
  std::vector<float> input(200);
  for (std::size_t i = 0; i < 200; ++i) {
    input[i] = static_cast<float>(i) * 0.01f - 1.0f;
  }

  std::vector<float> out_apply(rot.padded_dim());
  std::vector<float> out_inplace(rot.padded_dim());
  rot.apply(input, out_apply);

  std::memcpy(out_inplace.data(), input.data(), input.size() * sizeof(float));
  std::memset(out_inplace.data() + input.size(), 0,
              (rot.padded_dim() - input.size()) * sizeof(float));
  rot.apply_in_place(out_inplace);

  EXPECT_EQ(out_apply, out_inplace);
}

TEST(SrhtTest, PaddedDimsForBenchmarkDatasets) {
  EXPECT_EQ(SrhtRotation(200, 0).padded_dim(), 256u);
  EXPECT_EQ(SrhtRotation(1536, 0).padded_dim(), 2048u);
  EXPECT_EQ(SrhtRotation(3072, 0).padded_dim(), 4096u);
}
