#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <vector>

#include "vectorcache/transform/rotation.hpp"

using vectorcache::Rotation;

TEST(Rotation, BlockSizeRegimesPreserveNorm) {
  for (std::size_t dim : {8u, 200u, 256u, 768u, 1024u, 1536u}) {
    Rotation rot(dim);
    std::vector<float> row(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      row[i] = static_cast<float>(std::sin(0.1 * static_cast<double>(i) + 1.0));
    }
    double n0 = 0.0;
    for (float x : row) n0 += static_cast<double>(x) * x;
    n0 = std::sqrt(n0);
    for (float& x : row) x = static_cast<float>(x / n0);
    rot.apply(row);
    double n1 = 0.0;
    for (float x : row) n1 += static_cast<double>(x) * x;
    EXPECT_NEAR(std::sqrt(n1), 1.0, 1e-5) << "dim=" << dim;
  }
}

TEST(Rotation, ApplyScaledMatchesNormalizeThenApply) {
  constexpr std::size_t dim = 128;
  Rotation rot(dim);
  std::vector<float> src(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    src[i] = static_cast<float>(i + 1);
  }
  float nrm = 0.f;
  for (float x : src) nrm += x * x;
  nrm = std::sqrt(nrm);
  const float inv = 1.f / nrm;

  std::vector<float> a = src;
  for (float& x : a) x *= inv;
  rot.apply(a);

  std::vector<float> b(dim), scratch(dim);
  rot.apply_scaled_into(src, inv, b, scratch);

  for (std::size_t i = 0; i < dim; ++i) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(a[i]), std::bit_cast<std::uint32_t>(b[i])) << i;
  }
}

TEST(Rotation, Deterministic) {
  Rotation r1(64);
  Rotation r2(64);
  std::vector<float> x(64, 0.5f);
  std::vector<float> y = x;
  r1.apply(x);
  r2.apply(y);
  for (std::size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(std::bit_cast<std::uint32_t>(x[i]), std::bit_cast<std::uint32_t>(y[i]));
  }
}
