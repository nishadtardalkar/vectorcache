#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::query {

/// Asymmetric inner-product score: sum_i q_rot[i] * centroid[code_i] for one DB vector.
float asymmetric_ip_score(std::span<const float> query_rotated,
                          std::span<const std::uint64_t> data_words,
                          const quantize::LloydMaxCodebook& codebook);

/// Score num_vectors contiguous packed codes against a rotated float query.
void asymmetric_ip_batch(std::span<const float> query_rotated,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, const quantize::LloydMaxCodebook& codebook,
                         std::span<float> out_scores);

}  // namespace vectorcache::query
