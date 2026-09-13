#include <gtest/gtest.h>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(StoreTest, UniqueParentsAndGroups) {
  const std::size_t padded = 16;
  const std::size_t l0_words = l0_words_per_vector(padded);
  ParentStore store(l0_words, padded);

  const std::vector<std::uint64_t> l0_a(l0_words, 0x11);
  const std::vector<std::uint64_t> l0_b(l0_words, 0x22);
  const std::vector<std::uint64_t> l0_c(l0_words, 0x33);

  store.push_vector(0x0001, l0_a, 0);
  store.push_vector(0x0001, l0_b, 1);
  store.push_vector(0x0002, l0_c, 2);

  EXPECT_EQ(store.unique_parent_count(), 2u);
  EXPECT_EQ(store.total_vectors(), 3u);

  const auto keys = store.unique_keys();
  ASSERT_EQ(keys.size(), 2u);
  EXPECT_EQ(keys[0], 0x0001);
  EXPECT_EQ(keys[1], 0x0002);

  const ParentGroup* g1 = store.group_for_key(0x0001);
  ASSERT_NE(g1, nullptr);
  EXPECT_EQ(g1->size(), 2u);
  EXPECT_EQ(g1->id_at(0), 0u);
  EXPECT_EQ(g1->id_at(1), 1u);
  EXPECT_EQ(g1->vector_l0(0)[0], 0x11u);
  EXPECT_EQ(g1->vector_l0(1)[0], 0x22u);

  const ParentGroup* g2 = store.group_for_key(0x0002);
  ASSERT_NE(g2, nullptr);
  EXPECT_EQ(g2->size(), 1u);
  EXPECT_EQ(g2->id_at(0), 2u);

  EXPECT_EQ(store.group_for_key(0x0004), nullptr);
}

TEST(StoreTest, PushPostingsCountsOnce) {
  const std::size_t padded = 16;
  const std::size_t l0_words = l0_words_per_vector(padded);
  ParentStore store(l0_words, padded);
  const std::vector<std::uint64_t> l0(l0_words, 0xAB);
  const auto keys = parent_posting_keys(std::vector<float>(padded, 1.0f));

  store.push_postings(keys, l0, 7);
  EXPECT_EQ(store.total_vectors(), 1u);
  EXPECT_EQ(store.unique_parent_count(), PARENT_POSTINGS);

  for (const std::uint16_t key : keys) {
    const ParentGroup* g = store.group_for_key(key);
    ASSERT_NE(g, nullptr);
    EXPECT_EQ(g->size(), 1u);
    EXPECT_EQ(g->id_at(0), 7u);
  }
}

TEST(StoreTest, WithCapacityConstructs) {
  auto store = ParentStore::with_capacity(1, 16, 100);
  EXPECT_EQ(store.unique_parent_count(), 0u);
  EXPECT_EQ(store.l0_words_per_vec(), 1u);
  EXPECT_EQ(store.padded_dim(), 16u);
  EXPECT_EQ(store.total_vectors(), 0u);
}

TEST(StoreTest, RejectsChunkTooSmall) {
  EXPECT_THROW(ParentStore(1, 8), vectorcache::Error);
  EXPECT_THROW(ParentStore(1, 4), vectorcache::Error);
  EXPECT_NO_THROW(ParentStore(1, 16));
  EXPECT_NO_THROW(ParentStore(1, 24));
}
