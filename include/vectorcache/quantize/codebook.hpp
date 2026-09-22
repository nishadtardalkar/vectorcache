#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace vectorcache {

/// Lloyd-Max codebook for Beta((dim-1)/2, (dim-1)/2) on [-1, 1].
/// Returns (boundaries, centroids). Boundaries are f32 midpoints of f32 centroids.
std::pair<std::vector<float>, std::vector<float>> codebook(std::size_t bits, std::size_t dim);

}  // namespace vectorcache
