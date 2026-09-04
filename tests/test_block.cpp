#include <gtest/gtest.h>

#include "vectorcache/ingest/block.hpp"
#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(BlockLayoutTest, SimdFriendlyBenchmarkDims) {
  for (const std::size_t padded : {256u, 2048u, 4096u}) {
    const BlockLayout layout =
        make_block_layout(l1_words_per_vector(padded), l0_words_per_vector(padded), padded);
    EXPECT_TRUE(layout_is_simd_friendly(layout));

    LogicalBlock block(l1_words_per_vector(padded), l0_words_per_vector(padded), padded);
    block.push(std::vector<std::uint64_t>(l1_words_per_vector(padded), 0),
               std::vector<std::uint64_t>(l0_words_per_vector(padded), 0));
    EXPECT_EQ(block.layout().vector_l1_offset(0) % 64, 0u);
  }
}

TEST(BlockLayoutTest, GloveDimensions) {
  const BlockLayout layout = make_block_layout(1, 4, 256);
  EXPECT_EQ(layout.phys_block_bytes, BLOCK_SIZE * 8);
  EXPECT_EQ(layout.l0_phys_blocks, 4u);
  EXPECT_EQ(layout.l0_vecs_per_sub, 256u);
  EXPECT_EQ(layout.total_bytes, layout.phys_block_bytes * (1 + layout.l0_phys_blocks));
}

TEST(BlockLayoutTest, RegionOffsetsSequential) {
  const BlockLayout layout = make_block_layout(1, 4, 256);
  EXPECT_EQ(layout.l1_region_offset(), 0u);
  EXPECT_EQ(layout.l0_region_offset(), layout.phys_block_bytes);
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
  block.push(l1, l0);

  const auto bytes = block.bytes();
  const auto l1_slice = block.l1_slice();
  const auto l0_slice = block.l0_slice();

  EXPECT_EQ(reinterpret_cast<const std::byte*>(l1_slice.data()), bytes.data());
  EXPECT_EQ(reinterpret_cast<const std::byte*>(l0_slice.data()),
            bytes.data() + block.layout().l0_region_offset());
}

TEST(BlockTest, L1ContiguousLayout) {
  LogicalBlock block(1, 4, 256);
  const std::vector<std::uint64_t> l1_0 = {1};
  const std::vector<std::uint64_t> l1_1 = {3};
  const std::vector<std::uint64_t> l0(4, 0xFF);
  block.push(l1_0, l0);
  block.push(l1_1, l0);
  auto slice = block.l1_slice();
  ASSERT_EQ(slice.size(), 2u);
  EXPECT_EQ(slice[0], 1u);
  EXPECT_EQ(slice[1], 3u);
}

TEST(BlockTest, L0SubBlockBoundary) {
  LogicalBlock block(1, 4, 256);
  const std::vector<std::uint64_t> l1 = {0};
  const BlockLayout& layout = block.layout();

  for (std::size_t i = 0; i < 257; ++i) {
    const std::vector<std::uint64_t> l0_vec = {i, 0, 0, 0};
    block.push(l1, l0_vec);
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
  for (std::size_t i = 0; i < BLOCK_SIZE; ++i) {
    const std::vector<std::uint64_t> l1 = {i};
    block.push(l1, l0);
  }
  EXPECT_TRUE(block.is_full());
  EXPECT_EQ(block.len(), BLOCK_SIZE);
}

TEST(BlockTest, AsSliceAlias) {
  LogicalBlock block(1, 1, 8);
  const std::vector<std::uint64_t> l1 = {99};
  const std::vector<std::uint64_t> l0 = {0};
  block.push(l1, l0);
  EXPECT_EQ(block.as_slice()[0], 99u);
}

TEST(BlockTest, PerVectorAccessorsMatchSlices) {
  LogicalBlock block(1, 1, 8);
  const std::vector<std::uint64_t> l1_a = {1};
  const std::vector<std::uint64_t> l1_b = {2};
  const std::vector<std::uint64_t> l0_a = {0xAA};
  const std::vector<std::uint64_t> l0_b = {0xBB};
  block.push(l1_a, l0_a);
  block.push(l1_b, l0_b);

  EXPECT_EQ(block.vector_l1(0)[0], 1u);
  EXPECT_EQ(block.vector_l1(1)[0], 2u);
  EXPECT_EQ(block.vector_l0(0)[0], 0xAAu);
  EXPECT_EQ(block.vector_l0(1)[0], 0xBBu);
}
