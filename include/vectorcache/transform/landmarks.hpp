#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace vectorcache {

/// Orthonormal landmark matrix L (dim x m, column-major: L[d + j*dim]).
/// Built via QR of a seeded Gaussian matrix (thin Q's first m columns).
std::vector<float> make_ortho_landmarks(std::size_t dim, std::size_t m, std::uint64_t seed);

}  // namespace vectorcache
