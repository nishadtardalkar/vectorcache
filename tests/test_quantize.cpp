#include "vectorcache/quantize/quantize.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/dim_postings.hpp"

using namespace vectorcache::quantize;

TEST(QuantizeTest, L0AllPositive) {
  const std::vector<float> v(64, 1.0f);
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 64u);
  EXPECT_EQ(words.size(), 1u);
  EXPECT_EQ(words[0], ~0ull);
}

TEST(QuantizeTest, L0AllNegative) {
  const std::vector<float> v(64, -1.0f);
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 64u);
  EXPECT_EQ(words[0], 0ull);
}

TEST(QuantizeTest, L0MixedSigns) {
  std::vector<float> v(8, -1.0f);
  v[0] = 1.0f;
  v[3] = 1.0f;
  const auto [words, bits] = quantize_1dim_to_1bit(v);
  EXPECT_EQ(bits, 8u);
  EXPECT_EQ(words.size(), 1u);
  EXPECT_EQ(words[0] & 0xFFull, 0b00001001ull);
}

TEST(QuantizeTest, SupportKeyTopD) {
  std::vector<float> v(16, 0.1f);
  v[3] = 5.0f;
  v[7] = 4.0f;
  v[1] = 3.0f;
  v[9] = 2.0f;
  const auto key = quantize_support_key(v, 4);
  EXPECT_EQ(key.d, 4);
  EXPECT_EQ(key.dims[0], 1);
  EXPECT_EQ(key.dims[1], 3);
  EXPECT_EQ(key.dims[2], 7);
  EXPECT_EQ(key.dims[3], 9);
}

TEST(QuantizeTest, SupportKeyTieBreakHigherIndexWins) {
  std::vector<float> v(8, 1.0f);  // all |x|=1
  const auto key = quantize_support_key(v, 3);
  EXPECT_EQ(key.d, 3);
  // Highest indices win ties: 5,6,7
  EXPECT_EQ(key.dims[0], 5);
  EXPECT_EQ(key.dims[1], 6);
  EXPECT_EQ(key.dims[2], 7);
}

TEST(QuantizeTest, SupportHdAndNeighbors) {
  SupportKey a{};
  a.d = 2;
  a.dims[0] = 1;
  a.dims[1] = 3;
  SupportKey b = a;
  EXPECT_EQ(support_hd(a, b), 0);

  b.dims[1] = 4;
  // sorted: still need sorted keys
  b.dims[0] = 1;
  b.dims[1] = 4;
  EXPECT_EQ(support_hd(a, b), 2);

  std::size_t count = 0;
  for_each_hd2_neighbor(a, 8, [&](const SupportKey& n) {
    EXPECT_EQ(support_hd(a, n), 2);
    ++count;
    return true;
  });
  // d*(n-d) = 2*6 = 12
  EXPECT_EQ(count, 12u);
}

TEST(QuantizeTest, DimPostingHd2MatchesExistentialSubset) {
  SupportKey q{};
  q.d = 2;
  q.dims[0] = 1;
  q.dims[1] = 3;

  std::vector<SupportKey> keys;
  keys.push_back(q);
  SupportKey n1{};
  n1.d = 2;
  n1.dims[0] = 1;
  n1.dims[1] = 4;
  SupportKey n2{};
  n2.d = 2;
  n2.dims[0] = 3;
  n2.dims[1] = 5;
  SupportKey far{};
  far.d = 2;
  far.dims[0] = 0;
  far.dims[1] = 2;
  keys.push_back(n1);
  keys.push_back(n2);
  keys.push_back(far);

  auto key_less = [](const SupportKey& a, const SupportKey& b) {
    if (a.d != b.d) {
      return a.d < b.d;
    }
    return std::memcmp(a.dims, b.dims, a.d * sizeof(std::uint16_t)) < 0;
  };
  std::set<SupportKey, decltype(key_less)> expected(key_less);
  std::set<SupportKey, decltype(key_less)> from_postings(key_less);

  for_each_hd2_neighbor(q, 8, [&](const SupportKey& n) {
    for (const auto& k : keys) {
      if (k == n) {
        expected.insert(n);
      }
    }
    return true;
  });

  const auto idx = DimPostingIndex::build(keys, 8);
  idx.for_each_hd2(keys, q, [&](const SupportKey& n, std::uint32_t) {
    from_postings.insert(n);
    return true;
  });

  EXPECT_EQ(from_postings, expected);
  EXPECT_EQ(from_postings.size(), 2u);
}

TEST(QuantizeTest, DimPostingHd4) {
  SupportKey q{};
  q.d = 3;
  q.dims[0] = 0;
  q.dims[1] = 1;
  q.dims[2] = 2;

  SupportKey hd4{};
  hd4.d = 3;
  hd4.dims[0] = 0;
  hd4.dims[1] = 3;
  hd4.dims[2] = 4;
  EXPECT_EQ(support_hd(q, hd4), 4);

  SupportKey hd2{};
  hd2.d = 3;
  hd2.dims[0] = 0;
  hd2.dims[1] = 1;
  hd2.dims[2] = 5;
  EXPECT_EQ(support_hd(q, hd2), 2);

  std::vector<SupportKey> keys{q, hd4, hd2};
  const auto idx = DimPostingIndex::build(keys, 8);
  std::size_t n4 = 0;
  std::vector<std::uint8_t> visited(keys.size());
  idx.for_each_hd4(keys, q, visited, [&](const SupportKey& n, std::uint32_t) {
    EXPECT_EQ(support_hd(q, n), 4);
    EXPECT_EQ(n, hd4);
    ++n4;
    return true;
  });
  EXPECT_EQ(n4, 1u);
}

TEST(QuantizeTest, SupportKeyRejectsBadDepth) {
  EXPECT_THROW(quantize_support_key(std::vector<float>(4, 1.0f), 0), vectorcache::Error);
  EXPECT_THROW(quantize_support_key(std::vector<float>(4, 1.0f), 5), vectorcache::Error);
  EXPECT_THROW(quantize_support_key(std::vector<float>(3, 1.0f), 4), vectorcache::Error);
}
