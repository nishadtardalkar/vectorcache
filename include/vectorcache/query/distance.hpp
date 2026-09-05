#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

namespace vectorcache::query {

/// Sign-bit agreement score in [-1, 1] for num_bits valid bits in each word span.
float bit_agreement_score(std::span<const std::uint64_t> query_words,
                          std::span<const std::uint64_t> data_words, std::size_t num_bits);

/// Score num_vectors contiguous data vectors (data_words_per_vec u64 words each) against query.
void bit_agreement_batch(std::span<const std::uint64_t> query_words, std::size_t num_bits,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, std::span<float> out_scores);

/// Same as bit_agreement_batch but writes XOR popcount (disagreement bit counts).
/// Lower disagree is better; float score = (2*(num_bits-disagree) - num_bits) / num_bits.
void bit_agreement_batch_disagree(std::span<const std::uint64_t> query_words,
                                  std::size_t num_bits, std::span<const std::uint64_t> data_words,
                                  std::size_t data_words_per_vec, std::size_t num_vectors,
                                  std::span<std::uint32_t> out_disagree);

/// Convert disagreement bit count to agreement score in [-1, 1].
inline float score_from_disagree(std::uint32_t disagree, std::size_t num_bits) {
  const std::size_t agree = num_bits - static_cast<std::size_t>(disagree);
  return (2.0f * static_cast<float>(agree) - static_cast<float>(num_bits)) /
         static_cast<float>(num_bits);
}

}  // namespace vectorcache::query
