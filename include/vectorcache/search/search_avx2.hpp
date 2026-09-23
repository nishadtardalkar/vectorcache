#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/constants.hpp"

namespace vectorcache {

struct QueryLutView {
  const std::uint8_t* uint8_luts = nullptr;
  float scale = 1.f;
  float bias = 0.f;
};

void score_query_avx2_perm0(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                            std::span<const float> vec_scales, std::size_t n_byte_groups,
                            std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                            float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                            float& heap_min, std::size_t& heap_mi, float bias_corr,
                            const std::uint64_t* id_map = nullptr);

}  // namespace vectorcache
