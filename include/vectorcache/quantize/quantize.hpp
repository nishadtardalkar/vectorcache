#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

/// 4 orthogonal Hadamard basis vectors for 4D hyperplane projections.
inline constexpr std::array<std::array<float, 4>, 4> HADAMARD_4D = {{
    {{1.0f, 1.0f, 1.0f, 1.0f}},
    {{1.0f, -1.0f, 1.0f, -1.0f}},
    {{1.0f, 1.0f, -1.0f, -1.0f}},
    {{1.0f, -1.0f, -1.0f, 1.0f}},
}};

/// Number of u64 words needed to store L1 bitcodes for a vector of dim floats.
std::size_t l1_words_per_vector(std::size_t dim);

/// Number of L1 bits for a vector of dim floats.
std::size_t l1_bits_per_vector(std::size_t dim);

/// Number of u64 words needed to store L0 1dim→1bit bitcodes for a vector of dim floats.
std::size_t l0_words_per_vector(std::size_t dim);

/// Number of L0 bits for a vector of dim floats (one sign bit per dimension).
std::size_t l0_bits_per_vector(std::size_t dim);

/// Quantize a rotated vector to L1 4D→1bit bitcodes, writing into out.
/// Returns the number of valid bits. out.size() must equal l1_words_per_vector(dim).
std::size_t quantize_4d_to_1bit_into(std::span<const float> vector, std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_4d_to_1bit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_4d_to_1bit(
    std::span<const float> vector);

/// Quantize a rotated vector to L0 1dim→1bit bitcodes (sign per dimension), writing into out.
std::size_t quantize_1dim_to_1bit_into(std::span<const float> vector, std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_1dim_to_1bit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_1bit(
    std::span<const float> vector);

}  // namespace vectorcache::quantize
