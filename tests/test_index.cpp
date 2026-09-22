#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vectorcache/index.hpp"

TEST(Index, AddAndSearchReturnsIds) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 128;
  constexpr std::size_t nq = 4;
  constexpr std::size_t k = 5;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> db(n * dim);
  for (float& x : db) x = dist(rng);
  std::vector<float> queries(nq * dim);
  for (float& x : queries) x = dist(rng);

  vectorcache::TurboQuantIndex index(dim, 4);
  index.add(db);
  index.prepare();
  auto res = index.search(queries, k);
  ASSERT_EQ(res.nq, nq);
  ASSERT_EQ(res.k, k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_LT(res.ids[qi * k + j], n);
    }
    // Scores should be non-increasing.
    for (std::size_t j = 1; j < k; ++j) {
      EXPECT_GE(res.scores[qi * k + j - 1], res.scores[qi * k + j]);
    }
  }
}

TEST(Index, CalibrateBeforeAdd) {
  constexpr std::size_t dim = 32;
  std::vector<float> sample(1000 * dim, 0.1f);
  for (std::size_t i = 0; i < sample.size(); ++i) {
    sample[i] = static_cast<float>(std::sin(0.01 * static_cast<double>(i)));
  }
  vectorcache::TurboQuantIndex index(dim, 2);
  index.calibrate(sample);
  EXPECT_TRUE(index.has_calibration());
  index.add(sample);
  auto res = index.search(std::span<const float>(sample.data(), dim), 3);
  EXPECT_EQ(res.k, 3u);
}
