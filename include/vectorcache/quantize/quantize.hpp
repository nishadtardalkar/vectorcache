#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

/// Number of u64 words needed to store L0 1dim→1bit bitcodes for a vector of dim floats.
std::size_t l0_words_per_vector(std::size_t dim);

/// Number of L0 bits for a vector of dim floats (one sign bit per dimension).
std::size_t l0_bits_per_vector(std::size_t dim);

/// Quantize a rotated (srht_dim) vector to L0 1dim→1bit bitcodes, writing into out.
std::size_t quantize_1dim_to_1bit_into(std::span<const float> vector, std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_1dim_to_1bit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_1bit(
    std::span<const float> vector);

}  // namespace vectorcache::quantize
