#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/search/search_avx2.hpp"

namespace vectorcache {

/// Optional accumulators for VNNI kernel profiling (nanoseconds, all threads).
/// Activate with `vnni_profile_begin` around a sign (or other VNNI) scan.
struct VnniKernelProfile {
  std::atomic<std::uint64_t> vnni_ns{0};
  std::atomic<std::uint64_t> epilogue_ns{0};
};

void vnni_profile_begin(VnniKernelProfile* profile);
void vnni_profile_end();

/// Optional accumulators for the staged 4-bit shortlist kernel (nanoseconds).
struct ShortlistKernelProfile {
  std::atomic<std::uint64_t> gather_ns{0};
  std::atomic<std::uint64_t> alu_ns{0};
  std::atomic<std::uint64_t> epilogue_ns{0};
};

void shortlist_profile_begin(ShortlistKernelProfile* profile);
void shortlist_profile_end();

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

/// Identity copy of per-group `[hi_16 | lo_16]` LUTs (192×1; no quad reshape).
std::vector<std::uint8_t> split_lut_for_vnni(std::span<const std::uint8_t> uint8_luts,
                                            std::size_t n_byte_groups);

/// Score one query over 192×1 blocked codes with AVX-512 VNNI-LUT
/// (`vpshufb` + u32 accumulate). `lut.uint8_luts` from `split_lut_for_vnni`.
void score_query_vnni(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                      std::span<const float> vec_scales, std::size_t n_byte_groups,
                      std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                      std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                      std::size_t& heap_mi, float bias_corr,
                      const std::uint64_t* id_map = nullptr, float score_offset = 0.f,
                      const float* residual_norms = nullptr, float residual_scale = 0.f);

/// Sign-plane corpus scan specialized for `n_byte_groups == 32` (m=256 landmarks).
/// Weighted ±q nibble LUT (same math as `build_sign_lut`); int-first H99 epilogue
/// (no residual); software-pipelines the next 192-block under the flush.
/// Falls back to `score_query_vnni` when `n_byte_groups != 32`.
void score_sign_lut_m256(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                         std::span<const float> vec_scales, std::size_t n_byte_groups,
                         std::size_t n_vectors, std::size_t n_blocks, std::size_t k, float* heap_s,
                         std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min,
                         std::size_t& heap_mi, const std::uint64_t* id_map = nullptr);

/// Score up to 2 queries sharing one codes stream (192×1 VNNI-LUT).
/// `nq` must be in [1, 2]; larger batches are multiple passes at the caller.
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
                        const float* score_offsets = nullptr,
                        const float* residual_norms = nullptr,
                        const float* residual_scales = nullptr);

/// Same VNNI accumulate as `score_queries_vnni`, but writes `α · lut` into dense
/// `score_rows[qi][0..n_vectors)` (no top-k / collector). Dual-block + LUT hoist.
/// Used for staged sign shortlist → `nth_element` top-S after the scan.
void score_queries_vnni_to_rows(const std::uint8_t* const* split_luts, const float* lut_scales,
                                const float* lut_biases, std::size_t nq,
                                std::span<const std::uint8_t> blocked_codes,
                                std::span<const float> vec_scales, std::size_t n_byte_groups,
                                std::size_t n_vectors, std::size_t n_blocks,
                                float* const* score_rows);

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
                             std::size_t global_n_vectors = 0,
                             const float* residual_norms = nullptr, float residual_scale = 0.f,
                             const float* scale_resid = nullptr);

/// Score up to 10 queries with 4-bit permute-dot, one codes stream.
/// If `score_offsets` is non-null (length `nq`), adds `score_offsets[qi]` after α·LUT.
/// If `hit_bitmaps` is non-null, `hit_bitmaps[qi]` masks which vectors enter each heap.
/// Optional residual bound: score += residual_scales[qi] * residual_norms[vi].
/// If `scale_resid` is non-null, α/‖r‖ are AoS pairs (preferred flush load path).
void score_queries_permute_dot(const QueryPermuteDot* const* pds, std::size_t nq,
                               std::span<const std::uint8_t> blocked_codes,
                               std::span<const float> vec_scales, std::size_t n_byte_groups,
                               std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                               float* const* heap_s, std::uint64_t* const* heap_i,
                               std::size_t* heap_sz, float* heap_min, std::size_t* heap_mi,
                               const std::uint64_t* id_map = nullptr,
                               const float* score_offsets = nullptr,
                               const std::uint64_t* const* hit_bitmaps = nullptr,
                               std::size_t vec_offset = 0, std::size_t global_n_vectors = 0,
                               const float* residual_norms = nullptr,
                               const float* residual_scales = nullptr,
                               const float* scale_resid = nullptr);

/// Staged exact: 4-bit permute-dot over a shortlist of corpus ids.
/// Gathers `codes_flat` rows into W=64 SoA tiles (no 192-pad densify), scores all
/// byte-groups, then residual-aware flush. `cand_ids[i]` is the corpus vector id.
/// `scale_resid` is full-corpus AoS `[α, ‖r‖]` (length `2 * n_corpus`); if null,
/// uses `vec_scales` only (no residual term).
void score_shortlist_permute_dot(const QueryPermuteDot& pd,
                                 std::span<const std::uint8_t> codes_flat,
                                 std::size_t n_byte_groups,
                                 std::span<const std::uint64_t> cand_ids, std::size_t n_corpus,
                                 const float* scale_resid, const float* vec_scales, std::size_t k,
                                 float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                                 float& heap_min, std::size_t& heap_mi, float residual_scale);

}  // namespace vectorcache

