#include "vectorcache/ingest/store.hpp"

#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

#include "vectorcache/quantize/quantize.hpp"

using namespace vectorcache::ingest;
using namespace vectorcache::quantize;

TEST(StoreTest, PushAndFind) {
  constexpr std::size_t dim = 16;
  constexpr std::size_t top_d = 2;
  const std::size_t l0_words = l0_words_per_vector(dim);
  ParentStore store(l0_words, dim, dim, top_d);

  SupportKey k1 = quantize_support_key(std::vector<float>(dim, 1.0f), top_d);
  SupportKey k2{};
  k2.d = 2;
  k2.dims[0] = 0;
  k2.dims[1] = 1;

  std::vector<std::uint64_t> l0(l0_words, 1);
  store.push_vector(k1, l0, 10);
  store.push_vector(k1, l0, 11);
  store.push_vector(k2, l0, 20);

  EXPECT_EQ(store.total_vectors(), 3u);
  EXPECT_EQ(store.unique_parent_count(), 2u);

  const ParentGroup* g1 = store.find(k1);
  ASSERT_NE(g1, nullptr);
  EXPECT_EQ(g1->size(), 2u);
  EXPECT_EQ(g1->id_at(0), 10u);
  EXPECT_EQ(g1->id_at(1), 11u);

  const ParentGroup* g2 = store.find(k2);
  ASSERT_NE(g2, nullptr);
  EXPECT_EQ(g2->size(), 1u);
  EXPECT_EQ(g2->id_at(0), 20u);

  SupportKey missing{};
  missing.d = 2;
  missing.dims[0] = 2;
  missing.dims[1] = 3;
  EXPECT_EQ(store.find(missing), nullptr);

  EXPECT_EQ(store.input_dim(), dim);
  EXPECT_EQ(store.srht_dim(), dim);
  EXPECT_EQ(store.top_d(), top_d);
}
