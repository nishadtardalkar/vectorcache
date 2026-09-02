#include <gtest/gtest.h>

#include "vectorcache/ingest/block.hpp"

using namespace vectorcache::ingest;

TEST(BlockTest, ContiguousLayout) {
  VectorBlock block(2);
  const std::vector<std::uint64_t> c0 = {1, 2};
  const std::vector<std::uint64_t> c1 = {3, 4};
  block.push_l1(c0);
  block.push_l1(c1);
  auto slice = block.as_slice();
  ASSERT_EQ(slice.size(), 4u);
  EXPECT_EQ(slice[0], 1u);
  EXPECT_EQ(slice[1], 2u);
  EXPECT_EQ(slice[2], 3u);
  EXPECT_EQ(slice[3], 4u);
  EXPECT_EQ(block.len(), 2u);
}

TEST(BlockTest, FullBlockDetection) {
  VectorBlock block(1);
  for (std::size_t i = 0; i < BLOCK_SIZE; ++i) {
    const std::vector<std::uint64_t> code = {i};
    block.push_l1(code);
  }
  EXPECT_TRUE(block.is_full());
  EXPECT_EQ(block.len(), BLOCK_SIZE);
}
