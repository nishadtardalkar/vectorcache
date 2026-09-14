#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::quantize {

/// Dim → key_index posting lists for existential HD-neighbor enumeration.
class DimPostingIndex {
 public:
  DimPostingIndex() = default;

  static DimPostingIndex build(std::span<const SupportKey> keys, std::size_t input_dim);

  /// Visit existing keys at HD=2 from query. fn(key, key_idx) may return false to stop.
  /// Returns false if fn stopped early.
  template <typename Fn>
  bool for_each_hd2(std::span<const SupportKey> keys, const SupportKey& query, Fn&& fn) const {
    if (query.d < 1 || keys.size() != num_keys_) {
      return true;
    }
    const std::size_t d = query.d;
    std::uint16_t kept[kMaxSupportDepth];
    for (std::size_t drop_i = 0; drop_i < d; ++drop_i) {
      std::size_t w = 0;
      for (std::size_t j = 0; j < d; ++j) {
        if (j != drop_i) {
          kept[w++] = query.dims[j];
        }
      }
      if (!intersect_and_visit(keys, query, std::span<const std::uint16_t>(kept, w),
                               /*require_hd=*/2, fn)) {
        return false;
      }
    }
    return true;
  }

  /// Visit existing keys at HD=4 from query (deduped). fn(key, key_idx) may return false to stop.
  /// `visited` must have size >= keys.size(); cleared by this call.
  template <typename Fn>
  bool for_each_hd4(std::span<const SupportKey> keys, const SupportKey& query,
                    std::span<std::uint8_t> visited, Fn&& fn) const {
    if (query.d < 2 || keys.size() != num_keys_) {
      return true;
    }
    if (visited.size() < num_keys_) {
      throw Error("for_each_hd4: visited scratch too small");
    }
    std::memset(visited.data(), 0, num_keys_);
    const std::size_t d = query.d;
    std::uint16_t kept[kMaxSupportDepth];
    // All (d-2)-subsets of query dims (= C(d,2) drop pairs).
    for (std::size_t drop_a = 0; drop_a < d; ++drop_a) {
      for (std::size_t drop_b = drop_a + 1; drop_b < d; ++drop_b) {
        std::size_t w = 0;
        for (std::size_t j = 0; j < d; ++j) {
          if (j != drop_a && j != drop_b) {
            kept[w++] = query.dims[j];
          }
        }
        auto visit = [&](const SupportKey& key, std::uint32_t key_idx) {
          if (visited[key_idx]) {
            return true;
          }
          if (support_hd(query, key) != 4) {
            return true;
          }
          visited[key_idx] = 1;
          return fn(key, key_idx);
        };
        if (!intersect_and_visit_idx(keys, std::span<const std::uint16_t>(kept, w), visit)) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  template <typename Fn>
  bool intersect_and_visit(std::span<const SupportKey> keys, const SupportKey& query,
                           std::span<const std::uint16_t> dims, std::uint8_t require_hd,
                           Fn&& fn) const {
    return intersect_and_visit_idx(keys, dims, [&](const SupportKey& key, std::uint32_t key_idx) {
      if (key == query) {
        return true;
      }
      if (support_hd(query, key) != require_hd) {
        return true;
      }
      return fn(key, key_idx);
    });
  }

  template <typename VisitFn>
  bool intersect_and_visit_idx(std::span<const SupportKey> keys,
                               std::span<const std::uint16_t> dims, VisitFn&& visit) const {
    if (dims.empty()) {
      return true;
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < dims.size(); ++i) {
      if (postings_[dims[i]].size() < postings_[dims[best]].size()) {
        best = i;
      }
    }
    const auto& drive = postings_[dims[best]];
    for (std::uint32_t key_idx : drive) {
      bool ok = true;
      for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i == best) {
          continue;
        }
        const auto& list = postings_[dims[i]];
        if (!std::binary_search(list.begin(), list.end(), key_idx)) {
          ok = false;
          break;
        }
      }
      if (!ok) {
        continue;
      }
      if (!visit(keys[key_idx], key_idx)) {
        return false;
      }
    }
    return true;
  }

  std::vector<std::vector<std::uint32_t>> postings_;
  std::size_t num_keys_ = 0;
};

}  // namespace vectorcache::quantize
