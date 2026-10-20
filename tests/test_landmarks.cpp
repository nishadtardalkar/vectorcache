#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vectorcache/transform/landmarks.hpp"

TEST(Landmarks, OrthonormalColumns) {
  constexpr std::size_t dim = 128;
  constexpr std::size_t m = 96;
  auto L = vectorcache::make_ortho_landmarks(dim, m, /*seed=*/0);
  ASSERT_EQ(L.size(), dim * m);
  for (std::size_t i = 0; i < m; ++i) {
    double nrm = 0.0;
    for (std::size_t r = 0; r < dim; ++r) {
      const double v = L[r + i * dim];
      nrm += v * v;
    }
    EXPECT_NEAR(nrm, 1.0, 1e-5);
    for (std::size_t j = i + 1; j < m; ++j) {
      double dot = 0.0;
      for (std::size_t r = 0; r < dim; ++r) {
        dot += static_cast<double>(L[r + i * dim]) * L[r + j * dim];
      }
      EXPECT_NEAR(dot, 0.0, 1e-5) << i << "," << j;
    }
  }
}

TEST(Landmarks, Q4ProjectionCloseToFloat) {
  constexpr std::size_t dim = 128;
  constexpr std::size_t m = 96;
  constexpr std::size_t nq = 4;
  auto L = vectorcache::make_ortho_landmarks(dim, m, /*seed=*/1);
  auto Lq = vectorcache::quantize_landmarks(L, dim, m);
  ASSERT_EQ(Lq.m, m);
  ASSERT_EQ(Lq.dim, dim);
  ASSERT_GE(Lq.dim_pad, dim);
  ASSERT_EQ(Lq.codes.size(), (Lq.dim_pad / 2) * m);

  std::mt19937 rng(3);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> q(nq * dim);
  for (float& x : q) x = dist(rng);

  std::vector<float> exact(nq * m);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < m; ++j) {
      double s = 0.0;
      for (std::size_t i = 0; i < dim; ++i) {
        s += static_cast<double>(q[qi * dim + i]) * L[i + j * dim];
      }
      exact[qi * m + j] = static_cast<float>(s);
    }
  }

  std::vector<float> approx(nq * m, 0.f);
  vectorcache::project_onto_landmarks_q4(q.data(), nq, dim, Lq, approx.data(), m);

  double num = 0.0;
  double den = 0.0;
  for (std::size_t i = 0; i < exact.size(); ++i) {
    const double e = exact[i];
    const double a = approx[i];
    num += (e - a) * (e - a);
    den += e * e;
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  EXPECT_LT(rel, 0.15) << "rel_l2=" << rel;
}
