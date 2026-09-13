#pragma once

#include <array>
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

/// Number of equal contiguous chunks for parent dual-fold projection.
inline constexpr std::size_t PARENT_GROUPS = 8;

/// Packed parent key width: 2 bits per group × 8 groups.
inline constexpr std::size_t PARENT_KEY_BITS = 16;

/// Ingest multi-posts each vector under this many parent keys (2^PARENT_GROUPS).
inline constexpr std::size_t PARENT_POSTINGS = 256;

/// Max distinct parent keys (4^PARENT_GROUPS).
inline constexpr std::size_t PARENT_KEY_SPACE = 1u << PARENT_KEY_BITS;

/// Encode (fold, bit) into a 2-bit nibble: fold 0=F1, 1=F2; bit is 0 or 1.
inline constexpr std::uint16_t parent_nibble(std::uint8_t fold, std::uint8_t bit) {
  return static_cast<std::uint16_t>(((fold & 1u) << 1) | (bit & 1u));
}

/// Pack a nibble into parent key bits [2*group+1 : 2*group].
inline constexpr std::uint16_t parent_pack_nibble(std::size_t group, std::uint16_t nibble) {
  return static_cast<std::uint16_t>((nibble & 0x3u) << (2u * group));
}

/// F1/F2 sign bits per group (bit i = group i).
struct ParentFoldBits {
  std::uint8_t b1 = 0;
  std::uint8_t b2 = 0;
};

/// Project each chunk to 2D (all-ones, (1,-1,0,...)) and take sign bits.
/// Requires vector.size() % PARENT_GROUPS == 0 and chunk size >= 2.
ParentFoldBits quantize_parent_fold_bits(std::span<const float> vector);

/// All 256 ingest posting keys for a vector's fold bits.
std::array<std::uint16_t, PARENT_POSTINGS> parent_posting_keys(ParentFoldBits folds);

/// Convenience: fold bits → 256 posting keys.
std::array<std::uint16_t, PARENT_POSTINGS> parent_posting_keys(std::span<const float> vector);

/// Query parent key: per group pick fold with larger unit-normal margin, pack (fold, bit).
std::uint16_t quantize_parent_query_key(std::span<const float> vector);

/// Quantize a rotated vector to L0 1dim→1bit bitcodes (sign per dimension), writing into out.
std::size_t quantize_1dim_to_1bit_into(std::span<const float> vector, std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_1dim_to_1bit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_1bit(
    std::span<const float> vector);

}  // namespace vectorcache::quantize
