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
  index.set_score_threshold(-1e9f);  // threshold shortlist (no top-S heap)
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
  index.set_score_threshold(-1e9f);
  index.add(db);
  index.prepare();
  auto res = index.search(std::span<const float>(db.data(), dim), 1);
  ASSERT_EQ(res.k, 1u);
  EXPECT_EQ(res.ids[0], 0u);
  EXPECT_GT(res.scores[0], 0.9f);
}

TEST(Index, MultiQueryThresholdMatchesSingle) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 512;
  constexpr std::size_t nq = 8;
  constexpr std::size_t k = 8;

  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> db(n * dim);
  for (float& x : db) x = dist(rng);
  std::vector<float> queries(nq * dim);
  for (float& x : queries) x = dist(rng);

  vectorcache::LandmarkIndex index(dim, 4, /*n_landmarks=*/48, /*shortlist=*/128);
  index.set_score_threshold(-1e9f);
  index.add(db);
  index.prepare();

  auto batch = index.search(queries, k);
  ASSERT_EQ(batch.nq, nq);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    auto one = index.search(std::span<const float>(queries.data() + qi * dim, dim), k);
    ASSERT_EQ(one.nq, 1u);
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_EQ(batch.ids[qi * k + j], one.ids[j]) << "qi=" << qi << " j=" << j;
      EXPECT_NEAR(batch.scores[qi * k + j], one.scores[j], 1e-4f) << "qi=" << qi << " j=" << j;
    }
  }
}

TEST(Index, SeparateScanAndRerankBits) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 128;
  constexpr std::size_t nq = 2;
  constexpr std::size_t k = 3;

  std::mt19937 rng(99);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> db(n * dim);
  for (float& x : db) x = dist(rng);
  std::vector<float> queries(nq * dim);
  for (float& x : queries) x = dist(rng);

  vectorcache::LandmarkIndex index(dim, /*scan_bits=*/2, /*n_landmarks=*/32, /*shortlist=*/48,
                                   /*landmark_seed=*/0, /*rerank_bits=*/4);
  EXPECT_EQ(index.scan_bits(), 2u);
  EXPECT_EQ(index.rerank_bits(), 4u);
  EXPECT_EQ(index.bit_width(), 2u);
  index.set_score_threshold(-1e9f);
  index.add(db);
  index.prepare();
  auto res = index.search(queries, k);
  ASSERT_EQ(res.nq, nq);
  ASSERT_EQ(res.k, k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      EXPECT_LT(res.ids[qi * k + j], n);
    }
  }
}
