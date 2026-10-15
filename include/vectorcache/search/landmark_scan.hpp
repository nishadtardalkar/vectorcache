#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"

namespace vectorcache {

/// Pack row-major landmark rows (n * m) into blocked layout:
/// for each block of kBlock vectors: for each landmark dim, kBlock floats (SoA).
std::vector<float> pack_landmark_blocks(std::span<const float> rows, std::size_t n, std::size_t m);

/// Score one query landmark vector against blocked DB; keep top-`shortlist` (min-heap).
void landmark_shortlist_scan(std::span<const float> q_landmarks, std::size_t m,
                             std::span<const float> blocked, std::size_t n_vectors,
                             std::size_t n_blocks, std::size_t shortlist, float* heap_s,
                             std::uint64_t* heap_i, std::size_t& heap_sz, float& heap_min);

/// Quantized re-rank of candidate ids against flat full-dim codes; write top-k into out.
/// `codes_flat` is n_vectors × n_byte_groups; score = α · LUT(code) (asymmetric float query).
void quantized_rerank_topk(std::span<const std::uint8_t> lut_bytes, float lut_scale, float lut_bias,
                           std::span<const std::uint8_t> codes_flat, std::size_t n_byte_groups,
                           std::span<const float> scales, std::span<const std::uint64_t> cand_ids,
                           std::size_t n_cand, std::size_t k, float* out_scores,
                           std::uint64_t* out_ids);

}  // namespace vectorcache
