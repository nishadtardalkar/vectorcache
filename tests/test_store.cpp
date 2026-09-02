#include <gtest/gtest.h>

#include "vectorcache/ingest/store.hpp"

using namespace vectorcache::ingest;

TEST(StoreTest, TwoFullBlocksAndPartial) {
  const std::size_t words_per_vec = 1;
  const std::size_t l0_words = 4;
  const std::size_t padded = 256;
  BlockStore store(words_per_vec, l0_words, padded);
  const std::size_t total = BLOCK_SIZE * 2 + 3;
  const std::vector<std::uint64_t> l0(4, 0xAB);
  const std::vector<float> full(padded, 1.0f);

  for (std::size_t i = 0; i < total; ++i) {
    const std::vector<std::uint64_t> l1 = {i};
    store.push_vector(l1, l0, full);
  }

  EXPECT_EQ(store.block_count(), 2u);
  EXPECT_EQ(store.partial_block().len(), 3u);
  EXPECT_EQ(store.total_vectors(), total);

  const auto* block0 = store.get_block(0);
  ASSERT_NE(block0, nullptr);
  EXPECT_EQ(block0->len(), BLOCK_SIZE);
  EXPECT_EQ(block0->bytes().size(), store.layout().total_bytes);
  auto l1_slice = block0->l1_slice();
  EXPECT_EQ(l1_slice[0], 0u);

  const std::size_t offset = BLOCK_SIZE - 1;
  EXPECT_EQ(l1_slice[offset], static_cast<std::uint64_t>(BLOCK_SIZE - 1));

  const auto full_slice = block0->full_slice();
  EXPECT_FLOAT_EQ(full_slice[0], 1.0f);
  EXPECT_FLOAT_EQ(full_slice[(BLOCK_SIZE - 1) * padded], 1.0f);
}

TEST(StoreTest, WithCapacityPreallocatesContainers) {
  auto store = BlockStore::with_capacity(1, 4, 256, BLOCK_SIZE + 5);
  EXPECT_EQ(store.block_count(), 0u);
  EXPECT_EQ(store.l1_words_per_vec(), 1u);
  EXPECT_EQ(store.l0_words_per_vec(), 4u);
  EXPECT_EQ(store.padded_dim(), 256u);
}
