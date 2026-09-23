#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "vectorcache/search/topk_heap.hpp"

TEST(TopkHeap, ExactTopKWithTies) {
  constexpr std::size_t k = 64;
  constexpr std::size_t n = 500;

  std::vector<float> heap_s(k);
  std::vector<std::uint64_t> heap_i(k);
  std::size_t heap_sz = 0;
  float heap_min = 0.f;
  std::size_t heap_mi = 0;

  // Scores = id * 0.01f so top-k are the largest ids; a few equal scores to exercise ties.
  for (std::size_t i = 0; i < n; ++i) {
    float score = static_cast<float>(i) * 0.01f;
    if (i % 17 == 0) score = static_cast<float>((i / 17) * 17) * 0.01f;
    vectorcache::heap_push_or_replace(heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi, k,
                                      score, static_cast<std::uint64_t>(i));
  }

  ASSERT_EQ(heap_sz, k);
  std::vector<std::uint64_t> got(heap_i.begin(), heap_i.begin() + static_cast<std::ptrdiff_t>(k));
  std::sort(got.begin(), got.end());

  // Brute-force reference with same comparison: higher score wins; tie → lower id.
  std::vector<std::size_t> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::vector<float> scores(n);
  for (std::size_t i = 0; i < n; ++i) {
    scores[i] = static_cast<float>(i) * 0.01f;
    if (i % 17 == 0) scores[i] = static_cast<float>((i / 17) * 17) * 0.01f;
  }
  std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                    [&](std::size_t a, std::size_t b) {
                      if (scores[a] != scores[b]) return scores[a] > scores[b];
                      return a < b;
                    });
  std::vector<std::uint64_t> want(k);
  for (std::size_t i = 0; i < k; ++i) want[i] = static_cast<std::uint64_t>(order[i]);
  std::sort(want.begin(), want.end());
  EXPECT_EQ(got, want);
}
