#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace vectorcache {

/// True if (sa, ia) is strictly worse than (sb, ib) for a min-heap of top-k scores
/// (lower score is worse; on equal score, larger id is worse — matches prior rescan).
inline bool topk_worse(float sa, std::uint64_t ia, float sb, std::uint64_t ib) {
  if (sa != sb) return sa < sb;
  return ia > ib;
}

/// turbovec `rescan_min`: find worst entry (lowest score; ties → largest id).
inline void rescan_min(const float* heap_s, const std::uint64_t* heap_i, std::size_t k,
                       float& heap_min, std::size_t& heap_mi) {
  std::size_t mi = 0;
  for (std::size_t h = 1; h < k; ++h) {
    if (topk_worse(heap_s[h], heap_i[h], heap_s[mi], heap_i[mi])) {
      mi = h;
    }
  }
  heap_min = heap_s[mi];
  heap_mi = mi;
}

inline void topk_sift_down(float* heap_s, std::uint64_t* heap_i, std::size_t k, std::size_t i) {
  while (true) {
    const std::size_t l = 2 * i + 1;
    const std::size_t r = 2 * i + 2;
    std::size_t worst = i;
    if (l < k && topk_worse(heap_s[l], heap_i[l], heap_s[worst], heap_i[worst])) {
      worst = l;
    }
    if (r < k && topk_worse(heap_s[r], heap_i[r], heap_s[worst], heap_i[worst])) {
      worst = r;
    }
    if (worst == i) {
      break;
    }
    std::swap(heap_s[i], heap_s[worst]);
    std::swap(heap_i[i], heap_i[worst]);
    i = worst;
  }
}

inline void topk_heapify(float* heap_s, std::uint64_t* heap_i, std::size_t k) {
  if (k < 2) {
    return;
  }
  for (std::size_t i = k / 2; i-- > 0;) {
    topk_sift_down(heap_s, heap_i, k, i);
  }
}

/// Flat top-k. For small `k` (≤64) uses turbovec O(k) rescan; for larger shortlists
/// (staged sign plane, S≈16k) uses a binary min-heap so replaces are O(log k).
/// `heap_min` starts as `-infinity` or a cross-range prune seed. While filling,
/// only scores strictly above `heap_min` are kept (unseeded -inf → accept all
/// finite scores). Once full, replaces the worst when `score > heap_min`.
inline void heap_push_or_replace(float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                 float& heap_min, std::size_t& heap_mi, std::size_t k, float score,
                                 std::uint64_t id) {
  constexpr std::size_t kBinaryHeapMinK = 128;
  if (k < kBinaryHeapMinK) {
    if (heap_sz < k) {
      if (!(score > heap_min)) {
        return;
      }
      heap_s[heap_sz] = score;
      heap_i[heap_sz] = id;
      ++heap_sz;
      if (heap_sz == k) {
        rescan_min(heap_s, heap_i, k, heap_min, heap_mi);
      }
    } else if (score > heap_min) {
      heap_s[heap_mi] = score;
      heap_i[heap_mi] = id;
      rescan_min(heap_s, heap_i, k, heap_min, heap_mi);
    }
    return;
  }

  if (heap_sz < k) {
    if (!(score > heap_min)) {
      return;
    }
    heap_s[heap_sz] = score;
    heap_i[heap_sz] = id;
    ++heap_sz;
    if (heap_sz == k) {
      topk_heapify(heap_s, heap_i, k);
      heap_min = heap_s[0];
      heap_mi = 0;
    }
  } else if (score > heap_min) {
    heap_s[0] = score;
    heap_i[0] = id;
    topk_sift_down(heap_s, heap_i, k, 0);
    heap_min = heap_s[0];
    heap_mi = 0;
  }
}

inline std::uint64_t topk_map_id(const std::uint64_t* id_map, std::size_t local) {
  return id_map != nullptr ? id_map[local] : static_cast<std::uint64_t>(local);
}

}  // namespace vectorcache
