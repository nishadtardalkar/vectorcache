#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "vectorcache/index.hpp"

TEST(Index, AddAndSearchReturnsIds) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 256;
  constexpr std::size_t nq = 4;
  constexpr std::size_t k = 5;

  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> db(n * dim);
  for (float& x : db) x = dist(rng);
  std::vector<float> queries(nq * dim);
  for (float& x : queries) x = dist(rng);

  vectorcache::LandmarkIndex index(dim, 4, /*n_landmarks=*/48, /*shortlist=*/64);
  index.add(db);
  index.prepare();
  auto res = index.search(queries, k);
  ASSERT_EQ(res.nq, nq);
  ASSERT_EQ(res.k, k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_LT(res.ids[qi * k + j], n);
    }
    for (std::size_t j = 1; j < k; ++j) {
      EXPECT_GE(res.scores[qi * k + j - 1], res.scores[qi * k + j]);
    }
  }
}

TEST(Index, SelfQueryHighScore) {
  constexpr std::size_t dim = 128;
  constexpr std::size_t n = 64;
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> db(n * dim);
  for (float& x : db) x = dist(rng);

  vectorcache::LandmarkIndex index(dim, 4, 96, 32);
  index.add(db);
  index.prepare();
  auto res = index.search(std::span<const float>(db.data(), dim), 1);
  ASSERT_EQ(res.k, 1u);
  EXPECT_EQ(res.ids[0], 0u);
  EXPECT_GT(res.scores[0], 0.9f);
}
