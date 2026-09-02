#include <gtest/gtest.h>

#include "vectorcache/ingest/block.hpp"
#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(BlockLayoutTest, GloveDimensions) {
  const BlockLayout layout = make_block_layout(1, 4, 256);
  EXPECT_EQ(layout.phys_block_bytes, BLOCK_SIZE * 8);
  EXPECT_EQ(layout.l0_phys_blocks, 4u);
  EXPECT_EQ(layout.full_phys_blocks, 128u);
  EXPECT_EQ(layout.l0_vecs_per_sub, 256u);
  EXPECT_EQ(layout.full_vecs_per_sub, 8u);
  EXPECT_EQ(layout.total_bytes, layout.phys_block_bytes * 133);
}

TEST(BlockLayoutTest, RegionOffsetsSequential) {
  const BlockLayout layout = make_block_layout(1, 4, 256);
  EXPECT_EQ(layout.l1_region_offset(), 0u);
  EXPECT_EQ(layout.l0_region_offset(), layout.phys_block_bytes);
  EXPECT_EQ(layout.full_region_offset(), layout.phys_block_bytes * 5);
  EXPECT_EQ(layout.l0_sub_offset(1), layout.l0_region_offset() + layout.phys_block_bytes);
}

TEST(BlockTest, SingleBufferAllocation) {
  const BlockLayout layout = make_block_layout(1, 4, 256);
  LogicalBlock block(1, 4, 256);
  EXPECT_EQ(block.bytes().size(), layout.total_bytes);
  EXPECT_EQ(block.bytes().data(), block.layout().total_bytes > 0 ? block.bytes().data() : nullptr);
}

TEST(BlockTest, RegionContiguityInSingleBuffer) {
  LogicalBlock block(1, 1, 8);
  const std::vector<std::uint64_t> l1 = {0xAA};
  const std::vector<std::uint64_t> l0 = {0xBB};
  const std::vector<float> full(8, 1.5f);
  block.push(l1, l0, full);

  const auto bytes = block.bytes();
  const auto l1_slice = block.l1_slice();
  const auto l0_slice = block.l0_slice();
  const auto full_slice = block.full_slice();

  EXPECT_EQ(reinterpret_cast<const std::byte*>(l1_slice.data()), bytes.data());
  EXPECT_EQ(reinterpret_cast<const std::byte*>(l0_slice.data()),
            bytes.data() + block.layout().l0_region_offset());
  EXPECT_EQ(reinterpret_cast<const std::byte*>(full_slice.data()),
            bytes.data() + block.layout().full_region_offset());
}

TEST(BlockTest, L1ContiguousLayout) {
  LogicalBlock block(1, 4, 256);
  const std::vector<std::uint64_t> l1_0 = {1};
  const std::vector<std::uint64_t> l1_1 = {3};
  const std::vector<std::uint64_t> l0(4, 0xFF);
  const std::vector<float> full(256, 0.0f);
  block.push(l1_0, l0, full);
  block.push(l1_1, l0, full);
  auto slice = block.l1_slice();
  ASSERT_EQ(slice.size(), 2u);
  EXPECT_EQ(slice[0], 1u);
  EXPECT_EQ(slice[1], 3u);
}

TEST(BlockTest, L0SubBlockBoundary) {
  LogicalBlock block(1, 4, 256);
  const std::vector<std::uint64_t> l1 = {0};
  const std::vector<std::uint64_t> l0(4, 0);
  const std::vector<float> full(256, 0.0f);
  const BlockLayout& layout = block.layout();

  for (std::size_t i = 0; i < 257; ++i) {
    const std::vector<std::uint64_t> l0_vec = {i, 0, 0, 0};
    block.push(l1, l0_vec, full);
  }

  const auto bytes = block.bytes();
  const std::size_t off255 = layout.vector_l0_offset(255);
  const std::size_t off256 = layout.vector_l0_offset(256);
  EXPECT_EQ(off256 - off255, 4 * sizeof(std::uint64_t));
  EXPECT_EQ(*reinterpret_cast<const std::uint64_t*>(bytes.data() + off255), 255u);
  EXPECT_EQ(*reinterpret_cast<const std::uint64_t*>(bytes.data() + off256), 256u);
  EXPECT_EQ(off256, layout.l0_sub_offset(1));
}

TEST(BlockTest, FullBlockDetection) {
  LogicalBlock block(1, 1, 8);
  const std::vector<std::uint64_t> l0 = {0};
  const std::vector<float> full(8, 0.0f);
  for (std::size_t i = 0; i < BLOCK_SIZE; ++i) {
    const std::vector<std::uint64_t> l1 = {i};
    block.push(l1, l0, full);
  }
  EXPECT_TRUE(block.is_full());
  EXPECT_EQ(block.len(), BLOCK_SIZE);
}

TEST(BlockTest, AsSliceAlias) {
  LogicalBlock block(1, 1, 8);
  const std::vector<std::uint64_t> l1 = {99};
  const std::vector<std::uint64_t> l0 = {0};
  const std::vector<float> full(8, 0.0f);
  block.push(l1, l0, full);
  EXPECT_EQ(block.as_slice()[0], 99u);
}
