#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/query/query_config.hpp"

namespace vectorcache::query {

/// Sign-bit agreement score in [-1, 1] for num_bits valid bits in each word span.
float bit_agreement_score(std::span<const std::uint64_t> query_words,
                          std::span<const std::uint64_t> data_words, std::size_t num_bits);

/// Score num_vectors contiguous data vectors (data_words_per_vec u64 words each) against query.
void bit_agreement_batch(std::span<const std::uint64_t> query_words, std::size_t num_bits,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, std::span<float> out_scores);

#if VECTORCACHE_QUERY_DEPTH >= 3
float dot_f32(std::span<const float> a, std::span<const float> b);
#endif

}  // namespace vectorcache::query
