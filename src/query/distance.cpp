#include "vectorcache/query/distance.hpp"

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::query {

float asymmetric_ip_score(std::span<const float> query_rotated,
                          std::span<const std::uint64_t> data_words,
                          const quantize::LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  if (query_rotated.size() != dim) {
    throw Error("asymmetric_ip_score: query dim mismatch");
  }
  if (data_words.size() < quantize::l0_words_per_vector(dim, bits)) {
    throw Error("asymmetric_ip_score: data words too small");
  }

  float score = 0.0f;
  for (std::size_t d = 0; d < dim; ++d) {
    const std::uint32_t code = quantize::unpack_code(data_words, d, bits);
    score += query_rotated[d] * codebook.centroid_at(code);
  }
  return score;
}

void asymmetric_ip_batch(std::span<const float> query_rotated,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, const quantize::LloydMaxCodebook& codebook,
                         std::span<float> out_scores) {
  if (out_scores.size() < num_vectors || num_vectors == 0) {
    throw Error("asymmetric_ip_batch: invalid sizes");
  }
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  if (query_rotated.size() != dim) {
    throw Error("asymmetric_ip_batch: query dim mismatch");
  }
  if (data_words_per_vec != quantize::l0_words_per_vector(dim, bits)) {
    throw Error("asymmetric_ip_batch: words_per_vec mismatch");
  }
  if (data_words.size() < num_vectors * data_words_per_vec) {
    throw Error("asymmetric_ip_batch: data buffer too small");
  }

  for (std::size_t v = 0; v < num_vectors; ++v) {
    out_scores[v] = asymmetric_ip_score(
        query_rotated,
        data_words.subspan(v * data_words_per_vec, data_words_per_vec), codebook);
  }
}

}  // namespace vectorcache::query
