#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"

namespace vectorcache {

inline std::size_t bitmap_words(std::size_t n_vectors) {
  return (n_vectors + 63u) / 64u;
}

inline void bitmap_set(std::uint64_t* bits, std::size_t vi) {
  bits[vi >> 6] |= (std::uint64_t{1} << (vi & 63u));
}

inline bool bitmap_test(const std::uint64_t* bits, std::size_t vi) {
  return (bits[vi >> 6] >> (vi & 63u)) & 1u;
}

/// 32-bit hit mask for vectors `[base, base+kFlushChunk)` (base multiple of 32).
inline std::uint32_t block_hit_mask(const std::uint64_t* bits, std::size_t base,
                                   std::size_t n_vectors) {
  if (base >= n_vectors) return 0;
  const std::size_t end = std::min(base + kFlushChunk, n_vectors);
  const std::size_t word = base >> 6;
  const std::size_t bit = base & 63u;
  std::uint32_t mask = static_cast<std::uint32_t>((bits[word] >> bit) & 0xFFFFFFFFu);
  if (end - base < kFlushChunk) {
    mask &= (std::uint32_t{1} << (end - base)) - 1u;
  }
  return mask;
}

/// True if any of the kChunksPerBlock flush windows in a 192-wide block has a hit.
inline bool block_any_hit(const std::uint64_t* bits, std::size_t base, std::size_t n_vectors) {
  for (std::size_t c = 0; c < kChunksPerBlock; ++c) {
    if (block_hit_mask(bits, base + c * kFlushChunk, n_vectors) != 0) {
      return true;
    }
  }
  return false;
}

/// Pack row-major landmark rows (n * m) into blocked layout:
/// for each block of kBlock vectors: for each landmark dim, kBlock floats (SoA).
std::vector<float> pack_landmark_blocks(std::span<const float> rows, std::size_t n, std::size_t m);

/// Pass-2: walk flat full-dim codes in vector order; skip empty bitmap blocks; score only
/// vectors with bit set; write top-k into out.
void quantized_rerank_from_bitmap(std::span<const std::uint8_t> lut_bytes, float lut_scale,
                                  float lut_bias, std::span<const std::uint8_t> codes_flat,
                                  std::size_t n_byte_groups, std::span<const float> scales,
                                  const std::uint64_t* hit_bitmap, std::size_t n_vectors,
                                  std::size_t k, float* out_scores, std::uint64_t* out_ids);

}  // namespace vectorcache
