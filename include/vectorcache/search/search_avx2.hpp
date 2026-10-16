#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/constants.hpp"

namespace vectorcache {

inline constexpr std::size_t kAvx2NqBatch = 4;

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
                            const std::uint64_t* id_map = nullptr, float score_offset = 0.f);

/// Fused multi-query AVX2 PERM0 (turbovec `search_multi_query_avx2`).
/// `nq` must be in [1, kAvx2NqBatch]. Codes are loaded once per block; each
/// query keeps its own LUT / heap. Optional `bias_corrs` / `score_offsets`
/// are length `nq` (or null → 0).
void score_queries_avx2_perm0(const QueryLutView* luts, std::size_t nq,
                              std::span<const std::uint8_t> blocked_codes,
                              std::span<const float> vec_scales, std::size_t n_byte_groups,
                              std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                              float* const* heap_s, std::uint64_t* const* heap_i,
                              std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                              const float* bias_corrs = nullptr,
                              const std::uint64_t* id_map = nullptr,
                              const float* score_offsets = nullptr);

/// Score blocks into a fixed buffer: keep score >= threshold until `*out_count == capacity`,
/// then stop. If `out_scores` is null, only count hits (no early-stop). No allocations.
void score_query_avx2_perm0_threshold_fill(QueryLutView lut,
                                           std::span<const std::uint8_t> blocked_codes,
                                           std::span<const float> vec_scales,
                                           std::size_t n_byte_groups, std::size_t n_vectors,
                                           std::size_t n_blocks, float threshold,
                                           std::size_t capacity, float* out_scores,
                                           std::uint64_t* out_ids, std::size_t* out_count,
                                           float bias_corr = 0.f,
                                           const std::uint64_t* id_map = nullptr,
                                           float score_offset = 0.f);

/// Fused multi-query (nq ≤ kAvx2NqBatch): codes loaded once per block into fixed per-query
/// buffers. `out_scores[i]` / `out_ids[i]` / `out_counts[i]` are per-query; stop a query when
/// full, and stop the whole batch when every query is full.
void score_queries_avx2_perm0_threshold_fill(const QueryLutView* luts, std::size_t nq,
                                             std::span<const std::uint8_t> blocked_codes,
                                             std::span<const float> vec_scales,
                                             std::size_t n_byte_groups, std::size_t n_vectors,
                                             std::size_t n_blocks, float threshold,
                                             std::size_t capacity, float* const* out_scores,
                                             std::uint64_t* const* out_ids, std::size_t* out_counts,
                                             const float* bias_corrs = nullptr,
                                             const std::uint64_t* id_map = nullptr,
                                             const float* score_offsets = nullptr);

}  // namespace vectorcache
