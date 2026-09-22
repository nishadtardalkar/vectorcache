#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/search/search_avx2.hpp"

namespace vectorcache {

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

}  // namespace vectorcache
