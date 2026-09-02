#include <gtest/gtest.h>

#include "vectorcache/openmp.hpp"

using namespace vectorcache;

TEST(OpenMP, ConfigureDefaultDoesNotThrow) {
  configure_openmp_threads(std::nullopt);
  EXPECT_GE(openmp_max_threads(), 1);
}

TEST(OpenMP, ExplicitThreadCount) {
  configure_openmp_threads(2);
  EXPECT_EQ(openmp_max_threads(), 2);
}
