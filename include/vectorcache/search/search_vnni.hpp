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
                      std::size_t& heap_mi, float bias_corr,
                      const std::uint64_t* id_map = nullptr, float score_offset = 0.f);

/// Score up to 10 queries sharing one codes stream (classic VNNI / split LUT).
/// `nq` must be in [1, 10]. For `nq==1` prefers the dual-block kernel.
/// `split_luts[qi]` already split; `lut_biases[qi]` should include any TQ+ bias.
/// If `score_offsets` is non-null (length `nq`), adds `score_offsets[qi]` after α·LUT.
void score_queries_vnni(const std::uint8_t* const* split_luts, const float* lut_scales,
                        const float* lut_biases, std::size_t nq,
                        std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t n_byte_groups,
                        std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                        float* const* heap_s, std::uint64_t* const* heap_i, std::size_t* heap_sz,
                        float* heap_min, std::size_t* heap_mi,
                        const std::uint64_t* id_map = nullptr,
                        const float* score_offsets = nullptr);

/// Score one query with 4-bit permute-dot (dual-block when possible).
/// If `hit_bitmap` is non-null, only vectors with bit set enter the heap (empty blocks skipped).
/// `vec_offset` / `global_n_vectors` map local block indices into a global hit bitmap (block tiling).
void score_query_permute_dot(const QueryPermuteDot& pd, std::span<const std::uint8_t> blocked_codes,
                             std::span<const float> vec_scales, std::size_t n_byte_groups,
                             std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                             float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                             float& heap_min, std::size_t& heap_mi,
                             const std::uint64_t* id_map = nullptr, float score_offset = 0.f,
                             const std::uint64_t* hit_bitmap = nullptr, std::size_t vec_offset = 0,
                             std::size_t global_n_vectors = 0);

/// Score up to 10 queries with 4-bit permute-dot, one codes stream.
/// If `score_offsets` is non-null (length `nq`), adds `score_offsets[qi]` after α·LUT.
/// If `hit_bitmaps` is non-null, `hit_bitmaps[qi]` masks which vectors enter each heap.
void score_queries_permute_dot(const QueryPermuteDot* const* pds, std::size_t nq,
                               std::span<const std::uint8_t> blocked_codes,
                               std::span<const float> vec_scales, std::size_t n_byte_groups,
                               std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                               float* const* heap_s, std::uint64_t* const* heap_i,
                               std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                               const std::uint64_t* id_map = nullptr,
                               const float* score_offsets = nullptr,
                               const std::uint64_t* const* hit_bitmaps = nullptr,
                               std::size_t vec_offset = 0, std::size_t global_n_vectors = 0);

/// Threshold shortlist: keep score >= threshold into a fixed buffer (capacity).
/// If `out_scores` is null, only count hits (no early-stop). Capacity 0 with non-null outs is a no-op.
/// Single-query path uses BLK=8 interleaved streams + depth-8 prefetch (turbovec nq=1).
void score_query_permute_dot_threshold_fill(const QueryPermuteDot& pd,
                                            std::span<const std::uint8_t> blocked_codes,
                                            std::span<const float> vec_scales,
                                            std::size_t n_byte_groups, std::size_t n_vectors,
                                            std::size_t n_blocks, float threshold,
                                            std::size_t capacity, float* out_scores,
                                            std::uint64_t* out_ids, std::size_t* out_count,
                                            const std::uint64_t* id_map = nullptr,
                                            float score_offset = 0.f);

void score_query_vnni_threshold_fill(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                                     std::span<const float> vec_scales, std::size_t n_byte_groups,
                                     std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                     std::size_t capacity, float* out_scores,
                                     std::uint64_t* out_ids, std::size_t* out_count,
                                     float bias_corr = 0.f, const std::uint64_t* id_map = nullptr,
                                     float score_offset = 0.f);

/// Multi-query threshold fill (nq ∈ [1, 10]): one codes stream shared across the batch.
/// `nq==1` uses the BLK=8 single-query kernel; `nq>1` uses BLK=1 (permute-dot PF=32;
/// classic VNNI: no software prefetch, matching turbovec).
void score_queries_permute_dot_threshold_fill(const QueryPermuteDot* const* pds, std::size_t nq,
                                              std::span<const std::uint8_t> blocked_codes,
                                              std::span<const float> vec_scales,
                                              std::size_t n_byte_groups, std::size_t n_vectors,
                                              std::size_t n_blocks, float threshold,
                                              std::size_t capacity, float* const* out_scores,
                                              std::uint64_t* const* out_ids, std::size_t* out_counts,
                                              const std::uint64_t* id_map = nullptr,
                                              const float* score_offsets = nullptr);

void score_queries_vnni_threshold_fill(const std::uint8_t* const* split_luts,
                                       const float* lut_scales, const float* lut_biases,
                                       std::size_t nq, std::span<const std::uint8_t> blocked_codes,
                                       std::span<const float> vec_scales, std::size_t n_byte_groups,
                                       std::size_t n_vectors, std::size_t n_blocks, float threshold,
                                       std::size_t capacity, float* const* out_scores,
                                       std::uint64_t* const* out_ids, std::size_t* out_counts,
                                       const std::uint64_t* id_map = nullptr,
                                       const float* score_offsets = nullptr);

}  // namespace vectorcache
