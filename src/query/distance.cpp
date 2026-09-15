#include "vectorcache/query/distance.hpp"

#include <utility>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::query {

namespace {

float asymmetric_ip_score_scalar(std::span<const float> query_rotated,
                                 std::span<const std::uint64_t> data_words,
                                 const quantize::LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  const auto centroids = codebook.centroids();

  float score = 0.0f;
  for (std::size_t d = 0; d < dim; ++d) {
    const std::uint32_t code = quantize::unpack_code(data_words, d, bits);
    score += query_rotated[d] * centroids[code];
  }
  return score;
}

void fill_group_lut(float* table, std::span<const float> query_slice,
                    std::span<const float> centroids, std::size_t bits,
                    std::size_t dims_in_group) {
  const std::uint32_t code_mask = (bits == 32) ? 0xffffffffu : ((1u << bits) - 1u);
  for (std::size_t b = 0; b < QueryLut::kEntries; ++b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < dims_in_group; ++i) {
      const std::uint32_t code =
          static_cast<std::uint32_t>((b >> (i * bits)) & code_mask);
      sum += query_slice[i] * centroids[code];
    }
    table[b] = sum;
  }
}

float score_lut_one(const QueryLut& lut, const std::uint8_t* bytes) {
  const std::size_t groups = lut.num_groups();
  float a0 = 0.0f;
  float a1 = 0.0f;
  float a2 = 0.0f;
  float a3 = 0.0f;
  std::size_t g = 0;
  for (; g + 4 <= groups; g += 4) {
    a0 += lut.table(g + 0)[bytes[g + 0]];
    a1 += lut.table(g + 1)[bytes[g + 1]];
    a2 += lut.table(g + 2)[bytes[g + 2]];
    a3 += lut.table(g + 3)[bytes[g + 3]];
  }
  float score = a0 + a1 + a2 + a3;
  for (; g < groups; ++g) {
    score += lut.table(g)[bytes[g]];
  }
  return score;
}

void score_lut_batch(const QueryLut& lut, const std::uint8_t* base, std::size_t bytes_per_vec,
                     std::size_t num_vectors, std::span<float> out_scores) {
  for (std::size_t v = 0; v < num_vectors; ++v) {
    const std::uint8_t* row = base + v * bytes_per_vec;
#if defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(row + bytes_per_vec), _MM_HINT_T0);
#else
    __builtin_prefetch(row + bytes_per_vec, 0, 3);
#endif
    out_scores[v] = score_lut_one(lut, row);
  }
}

float score_bit1_one(const float* delta, float base, const std::uint64_t* words,
                     std::size_t dim) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  std::size_t d = 0;
  std::size_t word_i = 0;
  for (; d + 64 <= dim; d += 64, ++word_i) {
    const std::uint64_t w = words[word_i];
    {
      const __mmask16 m0 = static_cast<__mmask16>(w & 0xFFFFu);
      const __mmask16 m1 = static_cast<__mmask16>((w >> 16) & 0xFFFFu);
      acc0 = _mm512_mask_add_ps(acc0, m0, acc0, _mm512_loadu_ps(delta + d));
      acc1 = _mm512_mask_add_ps(acc1, m1, acc1, _mm512_loadu_ps(delta + d + 16));
    }
    {
      const __mmask16 m2 = static_cast<__mmask16>((w >> 32) & 0xFFFFu);
      const __mmask16 m3 = static_cast<__mmask16>((w >> 48) & 0xFFFFu);
      acc0 = _mm512_mask_add_ps(acc0, m2, acc0, _mm512_loadu_ps(delta + d + 32));
      acc1 = _mm512_mask_add_ps(acc1, m3, acc1, _mm512_loadu_ps(delta + d + 48));
    }
  }
  float score = base + _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  if (d < dim) {
    const std::uint64_t w = words[word_i];
    for (; d < dim; ++d) {
      if ((w >> (d % 64)) & 1ull) {
        score += delta[d];
      }
    }
  }
  return score;
}

/// Interleave 4 DB rows so mask-adds and delta loads overlap.
void score_bit1_four(const float* delta, float base, const std::uint64_t* r0,
                     const std::uint64_t* r1, const std::uint64_t* r2, const std::uint64_t* r3,
                     std::size_t dim, float* out4) {
  __m512 a0 = _mm512_setzero_ps();
  __m512 a1 = _mm512_setzero_ps();
  __m512 b0 = _mm512_setzero_ps();
  __m512 b1 = _mm512_setzero_ps();
  __m512 c0 = _mm512_setzero_ps();
  __m512 c1 = _mm512_setzero_ps();
  __m512 d0 = _mm512_setzero_ps();
  __m512 d1 = _mm512_setzero_ps();

  std::size_t off = 0;
  std::size_t word_i = 0;
  for (; off + 64 <= dim; off += 64, ++word_i) {
    const std::uint64_t w0 = r0[word_i];
    const std::uint64_t w1 = r1[word_i];
    const std::uint64_t w2 = r2[word_i];
    const std::uint64_t w3 = r3[word_i];
    const __m512 v0 = _mm512_loadu_ps(delta + off);
    const __m512 v1 = _mm512_loadu_ps(delta + off + 16);
    const __m512 v2 = _mm512_loadu_ps(delta + off + 32);
    const __m512 v3 = _mm512_loadu_ps(delta + off + 48);

    a0 = _mm512_mask_add_ps(a0, static_cast<__mmask16>(w0 & 0xFFFFu), a0, v0);
    b0 = _mm512_mask_add_ps(b0, static_cast<__mmask16>(w1 & 0xFFFFu), b0, v0);
    c0 = _mm512_mask_add_ps(c0, static_cast<__mmask16>(w2 & 0xFFFFu), c0, v0);
    d0 = _mm512_mask_add_ps(d0, static_cast<__mmask16>(w3 & 0xFFFFu), d0, v0);

    a1 = _mm512_mask_add_ps(a1, static_cast<__mmask16>((w0 >> 16) & 0xFFFFu), a1, v1);
    b1 = _mm512_mask_add_ps(b1, static_cast<__mmask16>((w1 >> 16) & 0xFFFFu), b1, v1);
    c1 = _mm512_mask_add_ps(c1, static_cast<__mmask16>((w2 >> 16) & 0xFFFFu), c1, v1);
    d1 = _mm512_mask_add_ps(d1, static_cast<__mmask16>((w3 >> 16) & 0xFFFFu), d1, v1);

    a0 = _mm512_mask_add_ps(a0, static_cast<__mmask16>((w0 >> 32) & 0xFFFFu), a0, v2);
    b0 = _mm512_mask_add_ps(b0, static_cast<__mmask16>((w1 >> 32) & 0xFFFFu), b0, v2);
    c0 = _mm512_mask_add_ps(c0, static_cast<__mmask16>((w2 >> 32) & 0xFFFFu), c0, v2);
    d0 = _mm512_mask_add_ps(d0, static_cast<__mmask16>((w3 >> 32) & 0xFFFFu), d0, v2);

    a1 = _mm512_mask_add_ps(a1, static_cast<__mmask16>((w0 >> 48) & 0xFFFFu), a1, v3);
    b1 = _mm512_mask_add_ps(b1, static_cast<__mmask16>((w1 >> 48) & 0xFFFFu), b1, v3);
    c1 = _mm512_mask_add_ps(c1, static_cast<__mmask16>((w2 >> 48) & 0xFFFFu), c1, v3);
    d1 = _mm512_mask_add_ps(d1, static_cast<__mmask16>((w3 >> 48) & 0xFFFFu), d1, v3);
  }

  out4[0] = base + _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
  out4[1] = base + _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
  out4[2] = base + _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
  out4[3] = base + _mm512_reduce_add_ps(_mm512_add_ps(d0, d1));
}

void score_bit1_batch(const QueryLut& lut, const std::uint64_t* data, std::size_t words_per_vec,
                      std::size_t num_vectors, std::span<float> out_scores) {
  const float* delta = lut.bit1_delta();
  const float base = lut.bit1_base();
  const std::size_t dim = lut.dim();
  std::size_t v = 0;
  for (; v + 4 <= num_vectors; v += 4) {
    const std::uint64_t* r0 = data + (v + 0) * words_per_vec;
    const std::uint64_t* r1 = data + (v + 1) * words_per_vec;
    const std::uint64_t* r2 = data + (v + 2) * words_per_vec;
    const std::uint64_t* r3 = data + (v + 3) * words_per_vec;
#if defined(_MSC_VER)
    _mm_prefetch(reinterpret_cast<const char*>(r0 + 4 * words_per_vec), _MM_HINT_T0);
#else
    __builtin_prefetch(r0 + 4 * words_per_vec, 0, 3);
#endif
    score_bit1_four(delta, base, r0, r1, r2, r3, dim, out_scores.data() + v);
  }
  for (; v < num_vectors; ++v) {
    out_scores[v] = score_bit1_one(delta, base, data + v * words_per_vec, dim);
  }
}

void asymmetric_ip_batch_scalar(std::span<const float> query_rotated,
                                std::span<const std::uint64_t> data_words,
                                std::size_t data_words_per_vec, std::size_t num_vectors,
                                const quantize::LloydMaxCodebook& codebook,
                                std::span<float> out_scores) {
  for (std::size_t v = 0; v < num_vectors; ++v) {
    out_scores[v] = asymmetric_ip_score_scalar(
        query_rotated, data_words.subspan(v * data_words_per_vec, data_words_per_vec), codebook);
  }
}

}  // namespace

bool lut_bits_supported(std::size_t bits) {
  return bits == 1 || bits == 2 || bits == 4 || bits == 8;
}

void build_query_lut(std::span<const float> query_rotated,
                     const quantize::LloydMaxCodebook& codebook, QueryLut& out) {
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  if (query_rotated.size() != dim) {
    throw Error("build_query_lut: query dim mismatch");
  }
  if (!lut_bits_supported(bits)) {
    out.clear();
    return;
  }
  if ((dim * bits) % 8 != 0) {
    out.clear();
    return;
  }

  const std::size_t dims_per_group = 8 / bits;
  const std::size_t num_groups = dim / dims_per_group;
  out.resize(num_groups, bits, dims_per_group, dim);

  const auto centroids = codebook.centroids();
  for (std::size_t g = 0; g < num_groups; ++g) {
    fill_group_lut(out.table(g), query_rotated.subspan(g * dims_per_group, dims_per_group),
                   centroids, bits, dims_per_group);
  }

  if (bits == 1 && (dim % 64) == 0) {
    const float c0 = centroids[0];
    const float c1 = centroids[1];
    const float dc = c1 - c0;
    AlignedVector<float> delta(dim);
    float base = 0.0f;
    for (std::size_t d = 0; d < dim; ++d) {
      base += query_rotated[d] * c0;
      delta[d] = query_rotated[d] * dc;
    }
    out.set_bit1_deltas(base, std::move(delta));
  }
}

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

  if (lut_bits_supported(bits) && (dim * bits) % 8 == 0) {
    QueryLut lut;
    build_query_lut(query_rotated, codebook, lut);
    return asymmetric_ip_score_lut(lut, data_words);
  }
  return asymmetric_ip_score_scalar(query_rotated, data_words, codebook);
}

float asymmetric_ip_score_lut(const QueryLut& lut, std::span<const std::uint64_t> data_words) {
  if (lut.empty()) {
    throw Error("asymmetric_ip_score_lut: empty lut");
  }
  const std::size_t need_words = (lut.bytes_per_vector() + 7) / 8;
  if (data_words.size() < need_words) {
    throw Error("asymmetric_ip_score_lut: data words too small");
  }
  if (lut.has_bit1_deltas()) {
    return score_bit1_one(lut.bit1_delta(), lut.bit1_base(), data_words.data(), lut.dim());
  }
  return score_lut_one(lut, reinterpret_cast<const std::uint8_t*>(data_words.data()));
}

void asymmetric_ip_batch(std::span<const float> query_rotated,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, const quantize::LloydMaxCodebook& codebook,
                         std::span<float> out_scores) {
  QueryLut lut;
  if (lut_bits_supported(codebook.bits())) {
    build_query_lut(query_rotated, codebook, lut);
  }
  asymmetric_ip_batch_lut(lut, data_words, data_words_per_vec, num_vectors, codebook,
                          query_rotated, out_scores);
}

void asymmetric_ip_batch_lut(const QueryLut& lut, std::span<const std::uint64_t> data_words,
                             std::size_t data_words_per_vec, std::size_t num_vectors,
                             const quantize::LloydMaxCodebook& codebook,
                             std::span<const float> query_rotated, std::span<float> out_scores) {
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

  if (lut.has_bit1_deltas() && lut.bits() == 1 && lut.dim() == dim) {
    score_bit1_batch(lut, data_words.data(), data_words_per_vec, num_vectors, out_scores);
    return;
  }

  if (lut.empty() || lut.bits() != bits || lut.bytes_per_vector() != (dim * bits) / 8) {
    asymmetric_ip_batch_scalar(query_rotated, data_words, data_words_per_vec, num_vectors,
                               codebook, out_scores);
    return;
  }

  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data_words.data());
  const std::size_t bytes_per_vec = data_words_per_vec * sizeof(std::uint64_t);
  score_lut_batch(lut, bytes, bytes_per_vec, num_vectors, out_scores);
}

}  // namespace vectorcache::query
