#include <gtest/gtest.h>

#include "vectorcache/ingest/store.hpp"

using namespace vectorcache::ingest;

TEST(StoreTest, TwoFullBlocksAndPartial) {
  const std::size_t words_per_vec = 2;
  BlockStore store(words_per_vec);
  const std::size_t total = BLOCK_SIZE * 2 + 3;

  for (std::size_t i = 0; i < total; ++i) {
    const std::vector<std::uint64_t> code = {i, i + 1};
    store.push_l1_codes(code);
  }

  EXPECT_EQ(store.block_count(), 2u);
  EXPECT_EQ(store.partial_block().len(), 3u);
  EXPECT_EQ(store.total_vectors(), total);

  const auto* block0 = store.get_block(0);
  ASSERT_NE(block0, nullptr);
  EXPECT_EQ(block0->len(), BLOCK_SIZE);
  auto slice = block0->as_slice();
  EXPECT_EQ(slice[0], 0u);
  EXPECT_EQ(slice[1], 1u);

  const std::size_t offset = (BLOCK_SIZE - 1) * words_per_vec;
  EXPECT_EQ(slice[offset], static_cast<std::uint64_t>(BLOCK_SIZE - 1));
  EXPECT_EQ(slice[offset + 1], static_cast<std::uint64_t>(BLOCK_SIZE));
}

TEST(StoreTest, WithCapacityPreallocatesContainers) {
  auto store = BlockStore::with_capacity(1, BLOCK_SIZE + 5);
  EXPECT_EQ(store.block_count(), 0u);
  EXPECT_EQ(store.l1_words_per_vec(), 1u);
}
