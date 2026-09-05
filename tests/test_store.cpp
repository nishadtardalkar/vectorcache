#include <gtest/gtest.h>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(StoreTest, UniqueParentsAndGroups) {
  const std::size_t padded = 8;
  const std::size_t l0_words = l0_words_per_vector(padded);
  ParentStore store(l0_words, padded);

  const std::vector<std::uint64_t> l0_a(l0_words, 0x11);
  const std::vector<std::uint64_t> l0_b(l0_words, 0x22);
  const std::vector<std::uint64_t> l0_c(l0_words, 0x33);

  store.push_vector(0b00000001, l0_a, 0);
  store.push_vector(0b00000001, l0_b, 1);
  store.push_vector(0b00000010, l0_c, 2);

  EXPECT_EQ(store.unique_parent_count(), 2u);
  EXPECT_EQ(store.total_vectors(), 3u);

  const auto keys = store.unique_keys();
  ASSERT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[0], 0b00000001);
  EXPECT_EQ(keys[1], 0b00000010);

  const ParentGroup* g1 = store.group_for_key(0b00000001);
  ASSERT_NE(g1, nullptr);
  EXPECT_EQ(g1->size(), 2u);
  EXPECT_EQ(g1->id_at(0), 0u);
  EXPECT_EQ(g1->id_at(1), 1u);
  EXPECT_EQ(g1->vector_l0(0)[0], 0x11u);
  EXPECT_EQ(g1->vector_l0(1)[0], 0x22u);

  const ParentGroup* g2 = store.group_for_key(0b00000010);
  ASSERT_NE(g2, nullptr);
  EXPECT_EQ(g2->size(), 1u);
  EXPECT_EQ(g2->id_at(0), 2u);

  EXPECT_EQ(store.group_for_key(0b00000100), nullptr);
}

TEST(StoreTest, WithCapacityConstructs) {
  auto store = ParentStore::with_capacity(1, 8, 100);
  EXPECT_EQ(store.unique_parent_count(), 0u);
  EXPECT_EQ(store.l0_words_per_vec(), 1u);
  EXPECT_EQ(store.padded_dim(), 8u);
  EXPECT_EQ(store.total_vectors(), 0u);
}

TEST(StoreTest, RejectsNonMultipleOfEightDim) {
  EXPECT_THROW(ParentStore(1, 4), vectorcache::Error);
}
