#include <gtest/gtest.h>

#include <bit>

#include "vectorcache/quantize/codebook.hpp"

TEST(Codebook, BoundariesAreF32Midpoints) {
  for (std::size_t bits : {2u, 3u, 4u}) {
    for (std::size_t dim : {8u, 128u, 200u, 768u}) {
      auto [boundaries, centroids] = vectorcache::codebook(bits, dim);
      ASSERT_EQ(centroids.size(), 1u << bits);
      ASSERT_EQ(boundaries.size(), (1u << bits) - 1);
      for (std::size_t i = 0; i < boundaries.size(); ++i) {
        const float expect = (centroids[i] + centroids[i + 1]) * 0.5f;
        EXPECT_EQ(std::bit_cast<std::uint32_t>(boundaries[i]),
                  std::bit_cast<std::uint32_t>(expect))
            << "bits=" << bits << " dim=" << dim << " i=" << i;
      }
      for (std::size_t i = 1; i < centroids.size(); ++i) {
        EXPECT_LT(centroids[i - 1], centroids[i]);
      }
    }
  }
}
