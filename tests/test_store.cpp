#include "vectorcache/ingest/store.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(StoreTest, PushAndLookup) {
  constexpr std::size_t dim = 16;
  const std::size_t l0_words = l0_words_per_vector(dim);
  VectorStore store(l0_words, dim, dim);

  std::vector<std::uint64_t> l0_a(l0_words, 1);
  std::vector<std::uint64_t> l0_b(l0_words, 2);
  store.push(10, l0_a);
  store.push(11, l0_a);
  store.push(20, l0_b);

  EXPECT_EQ(store.size(), 3u);
  EXPECT_EQ(store.id_at(0), 10u);
  EXPECT_EQ(store.id_at(1), 11u);
  EXPECT_EQ(store.id_at(2), 20u);
  EXPECT_EQ(store.vector_l0(0)[0], 1u);
  EXPECT_EQ(store.vector_l0(2)[0], 2u);
  EXPECT_EQ(store.l0_codes().size(), 3u * l0_words);
  EXPECT_EQ(store.input_dim(), dim);
  EXPECT_EQ(store.srht_dim(), dim);
}

TEST(StoreTest, WithCapacityReserves) {
  constexpr std::size_t dim = 64;
  const std::size_t l0_words = l0_words_per_vector(dim);
  auto store = VectorStore::with_capacity(l0_words, dim, dim, 100);
  EXPECT_TRUE(store.empty());
  std::vector<std::uint64_t> l0(l0_words, 0);
  store.push(0, l0);
  EXPECT_EQ(store.size(), 1u);
}
