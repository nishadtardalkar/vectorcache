#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <span>
#include <utility>
#include <vector>

namespace vectorcache::quantize {

/// Number of u64 words needed to store L0 1dim→1bit bitcodes for a vector of dim floats.
std::size_t l0_words_per_vector(std::size_t dim);

/// Number of L0 bits for a vector of dim floats (one sign bit per dimension).
std::size_t l0_bits_per_vector(std::size_t dim);

/// Max top-d depth stored in a SupportKey (compile-time cap).
inline constexpr std::size_t kMaxSupportDepth = 16;

/// Default top-d for support-key indexing.
inline constexpr std::size_t kDefaultSupportDepth = 4;

/// Fixed-weight support key: sorted top-d dimension indices (no signs).
struct SupportKey {
  std::uint8_t d = 0;
  std::uint16_t dims[kMaxSupportDepth] = {};

  bool empty() const { return d == 0; }

  bool operator==(const SupportKey& other) const {
    if (d != other.d) {
      return false;
    }
    return std::memcmp(dims, other.dims, static_cast<std::size_t>(d) * sizeof(std::uint16_t)) == 0;
  }
};

/// FNV-1a over the sorted dim list (stable, cheap for open-address maps).
inline std::uint64_t support_key_hash(const SupportKey& key) {
  std::uint64_t h = 14695981039346656037ull;
  for (std::uint8_t i = 0; i < key.d; ++i) {
    h ^= static_cast<std::uint64_t>(key.dims[i] & 0xFFu);
    h *= 1099511628211ull;
    h ^= static_cast<std::uint64_t>((key.dims[i] >> 8) & 0xFFu);
    h *= 1099511628211ull;
  }
  h ^= static_cast<std::uint64_t>(key.d);
  h *= 1099511628211ull;
  return h;
}

/// Hamming distance between two equal-weight support keys: 2*(d - |intersection|).
std::uint8_t support_hd(const SupportKey& a, const SupportKey& b);

/// Top-d dims by |x_i| on a normalized input_dim vector (no padding).
/// Tie-break: higher index wins when |x| equal.
SupportKey quantize_support_key(std::span<const float> vector, std::size_t d);

/// Enumerate all HD=2 neighbors (replace one included dim with one excluded).
/// Calls fn(neighbor) for each; fn may return false to stop early.
template <typename Fn>
void for_each_hd2_neighbor(const SupportKey& key, std::size_t input_dim, Fn&& fn) {
  if (key.d == 0 || input_dim < key.d) {
    return;
  }
  const std::size_t d = key.d;
  // Bitset membership: avoid O(d) scan per candidate add.
  constexpr std::size_t kStackWords = 64;  // covers dim <= 4096
  const std::size_t nwords = (input_dim + 63) / 64;
  std::uint64_t stack_bits[kStackWords];
  std::vector<std::uint64_t> heap_bits;
  std::uint64_t* bits = stack_bits;
  if (nwords > kStackWords) {
    heap_bits.assign(nwords, 0);
    bits = heap_bits.data();
  } else {
    std::memset(bits, 0, nwords * sizeof(std::uint64_t));
  }
  for (std::size_t j = 0; j < d; ++j) {
    const std::uint16_t dim = key.dims[j];
    bits[dim / 64] |= (std::uint64_t{1} << (dim % 64));
  }

  for (std::size_t drop_i = 0; drop_i < d; ++drop_i) {
    for (std::uint16_t add = 0; add < static_cast<std::uint16_t>(input_dim); ++add) {
      if ((bits[add / 64] >> (add % 64)) & 1ull) {
        continue;
      }
      SupportKey neigh{};
      neigh.d = key.d;
      std::size_t w = 0;
      for (std::size_t j = 0; j < d; ++j) {
        if (j == drop_i) {
          continue;
        }
        neigh.dims[w++] = key.dims[j];
      }
      neigh.dims[w++] = add;
      // Insertion-sort the d uint16s.
      for (std::size_t i = 1; i < d; ++i) {
        const std::uint16_t v = neigh.dims[i];
        std::size_t k = i;
        while (k > 0 && neigh.dims[k - 1] > v) {
          neigh.dims[k] = neigh.dims[k - 1];
          --k;
        }
        neigh.dims[k] = v;
      }
      if (!fn(neigh)) {
        return;
      }
    }
  }
}

/// Quantize a rotated (srht_dim) vector to L0 1dim→1bit bitcodes, writing into out.
std::size_t quantize_1dim_to_1bit_into(std::span<const float> vector, std::span<std::uint64_t> out);

/// Allocating wrapper around quantize_1dim_to_1bit_into.
std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_1bit(
    std::span<const float> vector);

}  // namespace vectorcache::quantize

namespace std {
template <>
struct hash<vectorcache::quantize::SupportKey> {
  std::size_t operator()(const vectorcache::quantize::SupportKey& key) const noexcept {
    return static_cast<std::size_t>(vectorcache::quantize::support_key_hash(key));
  }
};
}  // namespace std
