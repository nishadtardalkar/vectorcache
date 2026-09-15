#include <algorithm>
#include <cmath>
#include <cstdint>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/query/distance.hpp"
#include "vectorcache/query/fastscan.hpp"
#include "vectorcache/quantize/quantize.hpp"

using vectorcache::query::BlockedCodes;
using vectorcache::query::QueryLut;
using vectorcache::query::asymmetric_ip_batch_lut;
using vectorcache::query::build_query_lut;
using vectorcache::query::score_blocked_batch;
using vectorcache::quantize::LloydMaxCodebook;
using vectorcache::quantize::l0_words_per_vector;

namespace {

std::vector<std::uint64_t> pack_randomish(std::size_t dim, std::size_t bits, std::size_t n,
                                          std::uint64_t seed) {
  LloydMaxCodebook codebook(dim, bits);
  const std::size_t words = l0_words_per_vector(dim, bits);
  std::vector<std::uint64_t> codes(n * words, 0);
  std::vector<float> vec(dim);
  for (std::size_t v = 0; v < n; ++v) {
    for (std::size_t d = 0; d < dim; ++d) {
      const std::uint64_t x = seed + v * 1315423911u + d * 2654435761u;
      vec[d] = static_cast<float>((x % 2000) - 1000) * 0.001f;
    }
    const auto [packed, _] = vectorcache::quantize::quantize_1dim_to_nbit(vec, codebook);
    std::copy(packed.begin(), packed.end(), codes.begin() + static_cast<std::ptrdiff_t>(v * words));
  }
  return codes;
}

}  // namespace

TEST(FastScanPack, RoundTripBytes) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 1;
  constexpr std::size_t n = 40;  // spans a partial second block
  const std::size_t words = l0_words_per_vector(dim, bits);
  auto codes = pack_randomish(dim, bits, n, 42);

  BlockedCodes blocked;
  blocked.rebuild(codes, words, n, dim, bits);
  ASSERT_EQ(blocked.size(), n);
  ASSERT_EQ(blocked.n_blocks(), 2u);
  ASSERT_EQ(blocked.block_count(0), 32u);
  ASSERT_EQ(blocked.block_count(1), 8u);
  ASSERT_TRUE(blocked.has_bit1_words());

  std::vector<std::uint8_t> unpacked(blocked.num_groups());
  const auto* src = reinterpret_cast<const std::uint8_t*>(codes.data());
  const std::size_t bytes_per = words * sizeof(std::uint64_t);
  for (std::size_t v = 0; v < n; ++v) {
    blocked.unpack_vector_bytes(v, unpacked);
    for (std::size_t g = 0; g < blocked.num_groups(); ++g) {
      EXPECT_EQ(unpacked[g], src[v * bytes_per + g]) << "v=" << v << " g=" << g;
    }
  }
}

TEST(FastScanPack, Bit1ColumnMatchesRow) {
  constexpr std::size_t dim = 128;
  constexpr std::size_t bits = 1;
  constexpr std::size_t n = 35;
  const std::size_t words = l0_words_per_vector(dim, bits);
  auto codes = pack_randomish(dim, bits, n, 7);

  BlockedCodes blocked;
  blocked.rebuild(codes, words, n, dim, bits);
  for (std::size_t v = 0; v < n; ++v) {
    const std::size_t block = v / BlockedCodes::kBlock;
    const std::size_t vin = v % BlockedCodes::kBlock;
    for (std::size_t w = 0; w < words; ++w) {
      EXPECT_EQ(blocked.bit1_word_column(block, w)[vin], codes[v * words + w]);
    }
  }
}

TEST(FastScanScore, BlockedMatchesVectorMajorBits1) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 1;
  constexpr std::size_t n = 50;
  const std::size_t words = l0_words_per_vector(dim, bits);
  auto codes = pack_randomish(dim, bits, n, 99);
  LloydMaxCodebook codebook(dim, bits);

  std::vector<float> query(dim);
  for (std::size_t d = 0; d < dim; ++d) {
    query[d] = static_cast<float>((d % 17) - 8) * 0.05f;
  }

  QueryLut lut;
  build_query_lut(query, codebook, lut);
  ASSERT_TRUE(lut.has_bit1_deltas());

  std::vector<float> ref(n);
  asymmetric_ip_batch_lut(lut, codes, words, n, codebook, query, ref);

  BlockedCodes blocked;
  blocked.rebuild(codes, words, n, dim, bits);
  std::vector<float> got(n);
  for (std::size_t b = 0; b < blocked.n_blocks(); ++b) {
    const std::size_t count = blocked.block_count(b);
    score_blocked_batch(lut, blocked, b,
                        std::span<float>(got.data() + b * BlockedCodes::kBlock, count));
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]) << "i=" << i;
  }
}

TEST(FastScanScore, BlockedMatchesVectorMajorBits4) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 4;
  constexpr std::size_t n = 40;
  const std::size_t words = l0_words_per_vector(dim, bits);
  auto codes = pack_randomish(dim, bits, n, 123);
  LloydMaxCodebook codebook(dim, bits);

  std::vector<float> query(dim, 0.0f);
  for (std::size_t d = 0; d < dim; ++d) {
    query[d] = std::sin(0.1f * static_cast<float>(d));
  }

  QueryLut lut;
  build_query_lut(query, codebook, lut);
  ASSERT_FALSE(lut.empty());

  std::vector<float> ref(n);
  asymmetric_ip_batch_lut(lut, codes, words, n, codebook, query, ref);

  BlockedCodes blocked;
  blocked.rebuild(codes, words, n, dim, bits);
  std::vector<float> got(n);
  for (std::size_t b = 0; b < blocked.n_blocks(); ++b) {
    const std::size_t count = blocked.block_count(b);
    score_blocked_batch(lut, blocked, b,
                        std::span<float>(got.data() + b * BlockedCodes::kBlock, count));
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(got[i], ref[i], 1e-5f) << "i=" << i;
  }
}

TEST(FastScanScore, BlockedMatchesVectorMajorBits2) {
  constexpr std::size_t dim = 64;
  constexpr std::size_t bits = 2;
  constexpr std::size_t n = 33;
  const std::size_t words = l0_words_per_vector(dim, bits);
  auto codes = pack_randomish(dim, bits, n, 55);
  LloydMaxCodebook codebook(dim, bits);

  std::vector<float> query(dim, 0.02f);
  QueryLut lut;
  build_query_lut(query, codebook, lut);

  std::vector<float> ref(n);
  asymmetric_ip_batch_lut(lut, codes, words, n, codebook, query, ref);

  BlockedCodes blocked;
  blocked.rebuild(codes, words, n, dim, bits);
  std::vector<float> got(n);
  for (std::size_t b = 0; b < blocked.n_blocks(); ++b) {
    const std::size_t count = blocked.block_count(b);
    score_blocked_batch(lut, blocked, b,
                        std::span<float>(got.data() + b * BlockedCodes::kBlock, count));
  }
  for (std::size_t i = 0; i < n; ++i) {
    EXPECT_FLOAT_EQ(got[i], ref[i]) << "i=" << i;
  }
}
