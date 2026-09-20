#include "vectorcache/ingest/store.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(StoreTest, PushAndLookup) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t bits = 1;
  const std::size_t l0_words = l0_words_per_vector(dim, bits);
  VectorStore store(l0_words, dim, dim, bits);

  std::vector<std::uint64_t> l0_a(l0_words, 1);
  std::vector<std::uint64_t> l0_b(l0_words, 2);
  store.push(10, l0_a);
  store.push(11, l0_a);
  store.push(20, l0_b);

  EXPECT_EQ(store.size(), 3u);
  EXPECT_EQ(store.bits_per_dim(), bits);
  EXPECT_EQ(store.block_dims(), 1u);
  EXPECT_EQ(store.id_at(0), 10u);
  EXPECT_EQ(store.id_at(1), 11u);
  EXPECT_EQ(store.id_at(2), 20u);
  EXPECT_EQ(store.vector_l0(0)[0], 1u);
  EXPECT_EQ(store.vector_l0(2)[0], 2u);
  EXPECT_EQ(store.l0_codes().size(), 3u * l0_words);
  EXPECT_EQ(store.input_dim(), dim);
  EXPECT_EQ(store.srht_dim(), dim);
}

TEST(StoreTest, WithCapacityBlockDims) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 1;
  constexpr std::size_t block_dims = 2;
  const std::size_t l0_words = l0_words_per_vector(dim, bits, block_dims);
  auto store = VectorStore::with_capacity(l0_words, dim, dim, 10, bits, block_dims);
  EXPECT_EQ(store.block_dims(), block_dims);
  EXPECT_EQ(store.l0_words_per_vec(), 1u);
  std::vector<std::uint64_t> l0(l0_words, 0);
  store.push(0, l0);
  EXPECT_EQ(store.size(), 1u);
}

TEST(StoreTest, WithCapacityReservesMultiBit) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 2;
  const std::size_t l0_words = l0_words_per_vector(dim, bits);
  auto store = VectorStore::with_capacity(l0_words, dim, dim, 100, bits);
  EXPECT_TRUE(store.empty());
  EXPECT_EQ(store.bits_per_dim(), bits);
  EXPECT_EQ(store.l0_words_per_vec(), 2u);
  std::vector<std::uint64_t> l0(l0_words, 0);
  store.push(0, l0);
  EXPECT_EQ(store.size(), 1u);
}

TEST(StoreTest, CloneCopiesCodesWithoutBuckets) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t bits = 1;
  const std::size_t l0_words = l0_words_per_vector(dim, bits);
  VectorStore store(l0_words, dim, dim, bits);
  std::vector<std::uint64_t> l0_a(l0_words, 7);
  std::vector<std::uint64_t> l0_b(l0_words, 9);
  store.push(1, l0_a, 0.5f);
  store.push(2, l0_b, 1.5f);

  auto cloned = store.clone();
  EXPECT_EQ(cloned.size(), 2u);
  EXPECT_FALSE(cloned.has_buckets());
  EXPECT_EQ(cloned.id_at(0), 1u);
  EXPECT_EQ(cloned.id_at(1), 2u);
  EXPECT_EQ(cloned.scale_at(0), 0.5f);
  EXPECT_EQ(cloned.scale_at(1), 1.5f);
  EXPECT_EQ(cloned.vector_l0(0)[0], 7u);
  EXPECT_EQ(cloned.vector_l0(1)[0], 9u);
  EXPECT_EQ(cloned.l0_words_per_vec(), l0_words);
  EXPECT_EQ(cloned.bits_per_dim(), bits);
  EXPECT_EQ(cloned.input_dim(), dim);
  EXPECT_EQ(cloned.srht_dim(), dim);

  // Mutating clone must not affect original.
  std::vector<std::uint64_t> l0_c(l0_words, 3);
  cloned.push(3, l0_c);
  EXPECT_EQ(store.size(), 2u);
  EXPECT_EQ(cloned.size(), 3u);
}
