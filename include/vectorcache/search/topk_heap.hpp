#pragma once

#include <cstddef>
#include <cstdint>

namespace vectorcache {

/// True if (sa, ia) is strictly worse than (sb, ib) for a min-heap of top-k scores
/// (lower score is worse; on equal score, larger id is worse — matches prior rescan).
inline bool topk_worse(float sa, std::uint64_t ia, float sb, std::uint64_t ib) {
  if (sa != sb) return sa < sb;
  return ia > ib;
}

inline void topk_sift_up(float* heap_s, std::uint64_t* heap_i, std::size_t idx) {
  while (idx > 0) {
    const std::size_t parent = (idx - 1) / 2;
    if (!topk_worse(heap_s[idx], heap_i[idx], heap_s[parent], heap_i[parent])) break;
    const float ts = heap_s[idx];
    const std::uint64_t ti = heap_i[idx];
    heap_s[idx] = heap_s[parent];
    heap_i[idx] = heap_i[parent];
    heap_s[parent] = ts;
    heap_i[parent] = ti;
    idx = parent;
  }
}

inline void topk_sift_down(float* heap_s, std::uint64_t* heap_i, std::size_t heap_sz,
                           std::size_t idx) {
  for (;;) {
    std::size_t best = idx;
    const std::size_t left = 2 * idx + 1;
    const std::size_t right = left + 1;
    if (left < heap_sz &&
        topk_worse(heap_s[left], heap_i[left], heap_s[best], heap_i[best])) {
      best = left;
    }
    if (right < heap_sz &&
        topk_worse(heap_s[right], heap_i[right], heap_s[best], heap_i[best])) {
      best = right;
    }
    if (best == idx) break;
    const float ts = heap_s[idx];
    const std::uint64_t ti = heap_i[idx];
    heap_s[idx] = heap_s[best];
    heap_i[idx] = heap_i[best];
    heap_s[best] = ts;
    heap_i[best] = ti;
    idx = best;
  }
}

/// Binary min-heap top-k. When full, root is the worst entry (`heap_min` / index 0).
inline void heap_push_or_replace(float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                 float& heap_min, std::size_t& heap_mi, std::size_t k, float score,
                                 std::uint64_t id) {
  if (heap_sz < k) {
    heap_s[heap_sz] = score;
    heap_i[heap_sz] = id;
    topk_sift_up(heap_s, heap_i, heap_sz);
    ++heap_sz;
    if (heap_sz == k) {
      heap_min = heap_s[0];
      heap_mi = 0;
    }
  } else if (score > heap_min) {
    heap_s[0] = score;
    heap_i[0] = id;
    topk_sift_down(heap_s, heap_i, heap_sz, 0);
    heap_min = heap_s[0];
    heap_mi = 0;
  }
}

inline std::uint64_t topk_map_id(const std::uint64_t* id_map, std::size_t local) {
  return id_map != nullptr ? id_map[local] : static_cast<std::uint64_t>(local);
}

}  // namespace vectorcache
