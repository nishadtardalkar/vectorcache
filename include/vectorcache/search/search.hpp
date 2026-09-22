#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

struct SearchResults {
  std::vector<float> scores;      // nq * k
  std::vector<std::uint64_t> ids;  // nq * k
  std::size_t k = 0;
  std::size_t nq = 0;
};

SearchResults search_flat(std::span<const float> queries, std::size_t nq, std::size_t dim,
                          std::size_t k, const Rotation& rotation,
                          std::span<const float> centroids, std::size_t bits,
                          std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                          std::span<const float> scales,
                          std::span<const float> tqplus_shift,
                          std::span<const float> tqplus_scale);

}  // namespace vectorcache
