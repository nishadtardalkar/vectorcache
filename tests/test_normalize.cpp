#include <gtest/gtest.h>

#include "vectorcache/transform/normalize.hpp"

using namespace vectorcache::transform;

TEST(NormalizeTest, UnitVectorUnchangedUpToScale) {
  std::vector<float> v = {3.0f, 4.0f};
  l2_normalize_in_place(v);
  EXPECT_NEAR(v[0], 0.6f, 1e-6f);
  EXPECT_NEAR(v[1], 0.8f, 1e-6f);
}

TEST(NormalizeTest, ZeroVectorStaysZero) {
  std::vector<float> v(4, 0.0f);
  l2_normalize_in_place(v);
  for (float x : v) EXPECT_EQ(x, 0.0f);
}
