#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/constants.hpp"
#include "vectorcache/search/search_avx2.hpp"

namespace vectorcache {

/// Same NQ batch as TurboVec `search_multi_query_avx512bw` / AVX2 PERM0.
inline constexpr std::size_t kAvx512bwNqBatch = kAvx2NqBatch;

/// Fused multi-query AVX-512BW PERM0 (TurboVec `search_multi_query_avx512bw`).
/// Paired 32-vector blocks in ZMM; odd tail via AVX2 body. `nq` in [1, 4].
void score_queries_avx512bw_perm0(const QueryLutView* luts, std::size_t nq,
                                  std::span<const std::uint8_t> blocked_codes,
                                  std::span<const float> vec_scales, std::size_t n_byte_groups,
                                  std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                                  float* const* heap_s, std::uint64_t* const* heap_i,
                                  std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                                  const float* bias_corrs = nullptr,
                                  const std::uint64_t* id_map = nullptr,
                                  const float* score_offsets = nullptr);

/// Threshold collect (fixed buffer / count-only). Same contract as AVX2 threshold_fill.
void score_queries_avx512bw_perm0_threshold_fill(
    const QueryLutView* luts, std::size_t nq, std::span<const std::uint8_t> blocked_codes,
    std::span<const float> vec_scales, std::size_t n_byte_groups, std::size_t n_vectors,
    std::size_t n_blocks, float threshold, std::size_t capacity, float* const* out_scores,
    std::uint64_t* const* out_ids, std::size_t* out_counts, const float* bias_corrs = nullptr,
    const std::uint64_t* id_map = nullptr, const float* score_offsets = nullptr);

}  // namespace vectorcache
