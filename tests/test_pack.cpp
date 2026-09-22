#include <gtest/gtest.h>

#include <numeric>
#include <vector>

#include "vectorcache/pack/pack.hpp"

TEST(Pack, VectorMajorRoundTrip) {
  std::vector<std::uint8_t> buf(vectorcache::kVmUnit);
  std::iota(buf.begin(), buf.end(), 0);
  auto orig = buf;
  vectorcache::vector_major_chunk(buf);
  EXPECT_NE(buf, orig);
  vectorcache::vector_major_to_seq_chunk(buf);
  EXPECT_EQ(buf, orig);
}

TEST(Pack, RepackSeqThenRead) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 4;
  constexpr std::size_t n = 40;
  const std::size_t bytes_per_row = bits * (dim / 8);
  std::vector<std::uint8_t> packed(n * bytes_per_row);
  for (std::size_t i = 0; i < packed.size(); ++i) {
    packed[i] = static_cast<std::uint8_t>(i * 17 + 3);
  }
  auto [blocked, n_blocks] = vectorcache::repack(packed, n, bits, dim);
  EXPECT_GT(n_blocks, 0u);
  EXPECT_EQ(blocked.size(), n_blocks * (dim / (8 / bits)) * vectorcache::kBlock);

  // Round-trip via sequential form.
  auto seq = vectorcache::repack_seq(packed, n, bits, dim);
  EXPECT_EQ(seq.size(), blocked.size());
}
