#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/search/search_avx2.hpp"

namespace vectorcache {

/// Per-query state for the 4-bit permute-dot kernel (turbovec `QueryPermuteDot`).
struct QueryPermuteDot {
  std::array<std::int8_t, 16> levels{};
  std::vector<std::int8_t> weights;
  std::int32_t zero = 0;
  float scale = 1.f;
  float bias = 0.f;
};

/// Quantize query row + shared 16-centroid codebook for permute-dot.
QueryPermuteDot build_permute_dot(std::span<const float> q_rot_row,
                                  std::span<const float> centroids, std::size_t dim);

/// Rearrange per-group `[hi_16 | lo_16]` LUTs into vector-major VNNI order:
/// per quad of groups, `[lo0|lo1|lo2|lo3][hi0|hi1|hi2|hi3]` (128 bytes).
std::vector<std::uint8_t> split_lut_for_vnni(std::span<const std::uint8_t> uint8_luts,
                                            std::size_t n_byte_groups);

/// Score one query over vector-major blocked codes with AVX-512 VNNI
/// (`vpermb` + `vpdpbusd`). `lut.uint8_luts` must already be split via
/// `split_lut_for_vnni`. Requires `n_byte_groups % 4 == 0`.
void score_query_vnni(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                      std::span<const float> vec_scales, std::size_t n_byte_groups,
                      std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                      std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                      std::size_t& heap_mi, float bias_corr);

/// Score up to 8 queries sharing one codes stream (classic VNNI / split LUT).
/// `nq` must be in [1, 8]. For `nq==1` prefers the dual-block kernel.
/// `split_luts[qi]` already split; `lut_biases[qi]` should include any TQ+ bias.
void score_queries_vnni(const std::uint8_t* const* split_luts, const float* lut_scales,
                        const float* lut_biases, std::size_t nq,
                        std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t n_byte_groups,
                        std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                        float* const* heap_s, std::uint64_t* const* heap_i, std::size_t* heap_sz,
                        float* heap_min, std::size_t* heap_mi);

/// Score one query with 4-bit permute-dot (dual-block when possible).
void score_query_permute_dot(const QueryPermuteDot& pd, std::span<const std::uint8_t> blocked_codes,
                             std::span<const float> vec_scales, std::size_t n_byte_groups,
                             std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                             float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                             float& heap_min, std::size_t& heap_mi);

/// Score up to 8 queries with 4-bit permute-dot, one codes stream.
void score_queries_permute_dot(const QueryPermuteDot* const* pds, std::size_t nq,
                               std::span<const std::uint8_t> blocked_codes,
                               std::span<const float> vec_scales, std::size_t n_byte_groups,
                               std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                               float* const* heap_s, std::uint64_t* const* heap_i,
                               std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi);

}  // namespace vectorcache
