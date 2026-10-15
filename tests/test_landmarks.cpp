#include <gtest/gtest.h>

#include <cmath>
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
