#include "vectorcache/query/distance.hpp"

#include <cstring>
#include <utility>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/query/fastscan.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::query {

namespace {

float asymmetric_ip_score_scalar(std::span<const float> query_rotated,
                                 std::span<const std::uint64_t> data_words,
                                 const quantize::LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  const std::size_t block_dims = codebook.block_dims();
  const std::size_t m = dim / block_dims;

  float score = 0.0f;
  for (std::size_t b = 0; b < m; ++b) {
    const std::uint32_t code = quantize::unpack_code(data_words, b, bits);
    const auto c = codebook.centroid(code);
    for (std::size_t j = 0; j < block_dims; ++j) {
      score += query_rotated[b * block_dims + j] * c[j];
    }
  }
  return score;
}

void fill_group_lut(float* table, std::span<const float> query_slice,
                    std::span<const float> centroids, std::size_t bits,
                    std::size_t codes_in_group, std::size_t block_dims) {
  const std::uint32_t code_mask = (bits == 32) ? 0xffffffffu : ((1u << bits) - 1u);
  for (std::size_t b = 0; b < QueryLut::kEntries; ++b) {
    float sum = 0.0f;
    for (std::size_t i = 0; i < codes_in_group; ++i) {
      const std::uint32_t code =
          static_cast<std::uint32_t>((b >> (i * bits)) & code_mask);
      const float* c = centroids.data() + static_cast<std::size_t>(code) * block_dims;
      for (std::size_t j = 0; j < block_dims; ++j) {
        sum += query_slice[i * block_dims + j] * c[j];
      }
    }
    table[b] = sum;
  }
}

/// bits=8, block_dims==1: table[k] = q0 * c[k].
void fill_group_lut_bits8_d1(float* table, float q0, const float* centroids) {
  const __m512 vq = _mm512_set1_ps(q0);
  for (std::size_t k = 0; k < QueryLut::kEntries; k += 16) {
    const __m512 c = _mm512_loadu_ps(centroids + k);
    _mm512_storeu_ps(table + k, _mm512_mul_ps(vq, c));
  }
}

/// bits=8, block_dims==2: table[k] = q0*c[k,0] + q1*c[k,1] via AVX-512 FMA.
void fill_group_lut_bits8_d2(float* table, float q0, float q1, const float* centroids) {
  const __m512 vq0 = _mm512_set1_ps(q0);
  const __m512 vq1 = _mm512_set1_ps(q1);
  const __m512i pr_x = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 0, 2, 4, 6, 8, 10, 12, 14);
  const __m512i pr_y = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 1, 3, 5, 7, 9, 11, 13, 15);
  for (std::size_t k = 0; k < QueryLut::kEntries; k += 8) {
    const __m512 xy = _mm512_loadu_ps(centroids + k * 2);
    const __m512 xs = _mm512_permutexvar_ps(pr_x, xy);
    const __m512 ys = _mm512_permutexvar_ps(pr_y, xy);
    const __m512 dots = _mm512_fmadd_ps(vq0, xs, _mm512_mul_ps(vq1, ys));
    _mm256_storeu_ps(table + k, _mm512_castps512_ps256(dots));
  }
}

/// bits=4, block_dims==1: table[b] = q0*c[lo] + q1*c[hi].
void fill_group_lut_bits4_d1(float* table, float q0, float q1, const float* centroids) {
  alignas(64) float lo_dots[16];
  alignas(64) float hi_dots[16];
  const __m512 c = _mm512_loadu_ps(centroids);
  _mm512_store_ps(lo_dots, _mm512_mul_ps(_mm512_set1_ps(q0), c));
  _mm512_store_ps(hi_dots, _mm512_mul_ps(_mm512_set1_ps(q1), c));
  for (std::size_t b = 0; b < QueryLut::kEntries; ++b) {
    table[b] = lo_dots[b & 0x0Fu] + hi_dots[b >> 4];
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
                     std::size_t num_codes) {
  __m512 acc0 = _mm512_setzero_ps();
  __m512 acc1 = _mm512_setzero_ps();
  std::size_t d = 0;
  std::size_t word_i = 0;
  for (; d + 64 <= num_codes; d += 64, ++word_i) {
    const std::uint64_t w = words[word_i];
    {
      const __mmask16 m0 = static_cast<__mmask16>(w & 0xFFFFu);
      const __mmask16 m1 = static_cast<__mmask16>((w >> 16) & 0xFFFFu);
      acc0 = _mm512_mask_add_ps(acc0, m0, acc0, _mm512_load_ps(delta + d));
      acc1 = _mm512_mask_add_ps(acc1, m1, acc1, _mm512_load_ps(delta + d + 16));
    }
    {
      const __mmask16 m2 = static_cast<__mmask16>((w >> 32) & 0xFFFFu);
      const __mmask16 m3 = static_cast<__mmask16>((w >> 48) & 0xFFFFu);
      acc0 = _mm512_mask_add_ps(acc0, m2, acc0, _mm512_load_ps(delta + d + 32));
      acc1 = _mm512_mask_add_ps(acc1, m3, acc1, _mm512_load_ps(delta + d + 48));
    }
  }
  float score = base + _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
  if (d < num_codes) {
    const std::uint64_t w = words[word_i];
    for (; d < num_codes; ++d) {
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
                     std::size_t num_codes, float* out4) {
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
  for (; off + 64 <= num_codes; off += 64, ++word_i) {
    const std::uint64_t w0 = r0[word_i];
    const std::uint64_t w1 = r1[word_i];
    const std::uint64_t w2 = r2[word_i];
    const std::uint64_t w3 = r3[word_i];
    const __m512 v0 = _mm512_load_ps(delta + off);
    const __m512 v1 = _mm512_load_ps(delta + off + 16);
    const __m512 v2 = _mm512_load_ps(delta + off + 32);
    const __m512 v3 = _mm512_load_ps(delta + off + 48);

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
  const std::size_t num_codes = lut.num_codes();
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
    score_bit1_four(delta, base, r0, r1, r2, r3, num_codes, out_scores.data() + v);
  }
  for (; v < num_vectors; ++v) {
    out_scores[v] = score_bit1_one(delta, base, data + v * words_per_vec, num_codes);
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
  const std::size_t block_dims = codebook.block_dims();
  if (query_rotated.size() != dim) {
    throw Error("build_query_lut: query dim mismatch");
  }
  if (!lut_bits_supported(bits)) {
    out.clear();
    return;
  }
  if (dim % block_dims != 0) {
    out.clear();
    return;
  }
  const std::size_t m = dim / block_dims;
  if ((m * bits) % 8 != 0) {
    out.clear();
    return;
  }

  const std::size_t codes_per_group = 8 / bits;
  const std::size_t num_groups = m / codes_per_group;
  out.resize(num_groups, bits, m);

  const auto centroids = codebook.centroids();
  for (std::size_t g = 0; g < num_groups; ++g) {
    const std::size_t q_off = g * codes_per_group * block_dims;
    const auto q_slice = query_rotated.subspan(q_off, codes_per_group * block_dims);
    if (bits == 8 && block_dims == 1) {
      fill_group_lut_bits8_d1(out.table(g), q_slice[0], centroids.data());
    } else if (bits == 8 && block_dims == 2) {
      fill_group_lut_bits8_d2(out.table(g), q_slice[0], q_slice[1], centroids.data());
    } else if (bits == 4 && block_dims == 1) {
      fill_group_lut_bits4_d1(out.table(g), q_slice[0], q_slice[1], centroids.data());
    } else {
      fill_group_lut(out.table(g), q_slice, centroids, bits, codes_per_group, block_dims);
    }
  }

  if (bits == 1 && (m % 64) == 0 && codebook.num_centroids() >= 2) {
    AlignedVector<float> delta(m);
    float base = 0.0f;
    const auto c0 = codebook.centroid(0);
    const auto c1 = codebook.centroid(1);
    for (std::size_t b = 0; b < m; ++b) {
      float base_b = 0.0f;
      float delta_b = 0.0f;
      for (std::size_t j = 0; j < block_dims; ++j) {
        const float q = query_rotated[b * block_dims + j];
        base_b += q * c0[j];
        delta_b += q * (c1[j] - c0[j]);
      }
      base += base_b;
      delta[b] = delta_b;
    }
    out.set_bit1_deltas(base, std::move(delta));
  }
}

float asymmetric_ip_score(std::span<const float> query_rotated,
                          std::span<const std::uint64_t> data_words,
                          const quantize::LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.srht_dim();
  const std::size_t bits = codebook.bits();
  const std::size_t block_dims = codebook.block_dims();
  if (query_rotated.size() != dim) {
    throw Error("asymmetric_ip_score: query dim mismatch");
  }
  if (data_words.size() < quantize::l0_words_per_vector(dim, bits, block_dims)) {
    throw Error("asymmetric_ip_score: data words too small");
  }

  const std::size_t m = dim / block_dims;
  if (lut_bits_supported(bits) && (m * bits) % 8 == 0) {
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
    return score_bit1_one(lut.bit1_delta(), lut.bit1_base(), data_words.data(), lut.num_codes());
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
  const std::size_t block_dims = codebook.block_dims();
  const std::size_t m = dim / block_dims;
  if (query_rotated.size() != dim) {
    throw Error("asymmetric_ip_batch: query dim mismatch");
  }
  if (data_words_per_vec != quantize::l0_words_per_vector(dim, bits, block_dims)) {
    throw Error("asymmetric_ip_batch: words_per_vec mismatch");
  }
  if (data_words.size() < num_vectors * data_words_per_vec) {
    throw Error("asymmetric_ip_batch: data buffer too small");
  }

  if (lut.has_bit1_deltas() && lut.bits() == 1 && lut.num_codes() == m) {
    score_bit1_batch(lut, data_words.data(), data_words_per_vec, num_vectors, out_scores);
    return;
  }

  if (lut.empty() || lut.bits() != bits || lut.bytes_per_vector() != (m * bits) / 8) {
    asymmetric_ip_batch_scalar(query_rotated, data_words, data_words_per_vec, num_vectors,
                               codebook, out_scores);
    return;
  }

  const auto* bytes = reinterpret_cast<const std::uint8_t*>(data_words.data());
  const std::size_t bytes_per_vec = data_words_per_vec * sizeof(std::uint64_t);
  score_lut_batch(lut, bytes, bytes_per_vec, num_vectors, out_scores);
}

namespace {

/// Score one block via transposed bits=1 mask-add (shared delta loads).
/// Accumulates across all words in SIMD registers before one reduce (matches vector-major).
void score_blocked_bit1(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                        std::span<float> out_scores) {
  const std::size_t count = blocked.block_count(block);
  const float* delta = lut.bit1_delta();
  const float base = lut.bit1_base();
  const std::size_t num_codes = lut.num_codes();
  const std::size_t words = blocked.words_per_vec();

  std::size_t vin = 0;
  for (; vin + 4 <= count; vin += 4) {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps();
    __m512 b1 = _mm512_setzero_ps();
    __m512 c0 = _mm512_setzero_ps();
    __m512 c1 = _mm512_setzero_ps();
    __m512 d0 = _mm512_setzero_ps();
    __m512 d1 = _mm512_setzero_ps();

    std::size_t off = 0;
    for (std::size_t word_i = 0; word_i < words && off + 64 <= num_codes; ++word_i, off += 64) {
      const std::uint64_t* col = blocked.bit1_word_column_unchecked(block, word_i);
      const std::uint64_t w0 = col[vin + 0];
      const std::uint64_t w1 = col[vin + 1];
      const std::uint64_t w2 = col[vin + 2];
      const std::uint64_t w3 = col[vin + 3];
      const __m512 v0 = _mm512_load_ps(delta + off);
      const __m512 v1 = _mm512_load_ps(delta + off + 16);
      const __m512 v2 = _mm512_load_ps(delta + off + 32);
      const __m512 v3 = _mm512_load_ps(delta + off + 48);

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

    out_scores[vin + 0] = base + _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    out_scores[vin + 1] = base + _mm512_reduce_add_ps(_mm512_add_ps(b0, b1));
    out_scores[vin + 2] = base + _mm512_reduce_add_ps(_mm512_add_ps(c0, c1));
    out_scores[vin + 3] = base + _mm512_reduce_add_ps(_mm512_add_ps(d0, d1));
  }
  for (; vin < count; ++vin) {
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    std::size_t off = 0;
    for (std::size_t word_i = 0; word_i < words && off + 64 <= num_codes; ++word_i, off += 64) {
      const std::uint64_t w = blocked.bit1_word_column_unchecked(block, word_i)[vin];
      const __m512 v0 = _mm512_load_ps(delta + off);
      const __m512 v1 = _mm512_load_ps(delta + off + 16);
      const __m512 v2 = _mm512_load_ps(delta + off + 32);
      const __m512 v3 = _mm512_load_ps(delta + off + 48);
      a0 = _mm512_mask_add_ps(a0, static_cast<__mmask16>(w & 0xFFFFu), a0, v0);
      a1 = _mm512_mask_add_ps(a1, static_cast<__mmask16>((w >> 16) & 0xFFFFu), a1, v1);
      a0 = _mm512_mask_add_ps(a0, static_cast<__mmask16>((w >> 32) & 0xFFFFu), a0, v2);
      a1 = _mm512_mask_add_ps(a1, static_cast<__mmask16>((w >> 48) & 0xFFFFu), a1, v3);
    }
    out_scores[vin] = base + _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
  }
}

/// Blocked float LUT: for each byte-group, accumulate lut[g][byte_v] across kBlock lanes.
void score_blocked_float_lut(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                             std::span<float> out_scores) {
  const std::size_t count = blocked.block_count(block);
  const std::size_t groups = lut.num_groups();
  alignas(64) float acc[BlockedCodes::kBlock] = {};

  for (std::size_t g = 0; g < groups; ++g) {
    const std::uint8_t* codes = blocked.group_bytes_unchecked(block, g);
    const float* table = lut.table(g);
#if defined(_MSC_VER)
    if (g + 1 < groups) {
      _mm_prefetch(reinterpret_cast<const char*>(blocked.group_bytes_unchecked(block, g + 1)),
                   _MM_HINT_T0);
    }
#else
    if (g + 1 < groups) {
      __builtin_prefetch(blocked.group_bytes_unchecked(block, g + 1), 0, 3);
    }
#endif
    std::size_t v = 0;
    for (; v + 16 <= count; v += 16) {
      const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + v));
      const __m512i idx = _mm512_cvtepu8_epi32(bytes);
      const __m512 vals = _mm512_i32gather_ps(idx, table, 4);
      const __m512 vacc = _mm512_loadu_ps(acc + v);
      _mm512_storeu_ps(acc + v, _mm512_add_ps(vacc, vals));
    }
    for (; v < count; ++v) {
      acc[v] += table[codes[v]];
    }
  }
  std::memcpy(out_scores.data(), acc, count * sizeof(float));
}

/// Peel additive 4-bit pair tables: L(n)+H(m) = table[(m<<4)|n] (exact).
void peel_nibble4_tables(const float* table, float* lo, float* hi) {
  const float t0 = table[0];
  lo[0] = 0.0f;
  hi[0] = t0;
  for (std::size_t n = 1; n < 16; ++n) {
    lo[n] = table[n] - t0;
    hi[n] = table[n << 4];
  }
}

/// bits=4 FastScan: nibble-split float LUTs + AVX-512 permute (exact).
void score_blocked_nibble4(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                           std::span<float> out_scores) {
  const std::size_t count = blocked.block_count(block);
  const std::size_t groups = lut.num_groups();
  alignas(64) float acc[BlockedCodes::kBlock] = {};
  alignas(64) float lo[16];
  alignas(64) float hi[16];
  const __m512i nibble_mask = _mm512_set1_epi32(0x0f);

  for (std::size_t g = 0; g < groups; ++g) {
    peel_nibble4_tables(lut.table(g), lo, hi);
    const std::uint8_t* codes = blocked.group_bytes_unchecked(block, g);
    const __m512 lut_lo = _mm512_load_ps(lo);
    const __m512 lut_hi = _mm512_load_ps(hi);

    std::size_t v = 0;
    for (; v + 16 <= count; v += 16) {
      const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + v));
      const __m512i b32 = _mm512_cvtepu8_epi32(bytes);
      const __m512i ilo = _mm512_and_si512(b32, nibble_mask);
      const __m512i ihi = _mm512_srli_epi32(b32, 4);
      const __m512 vlo = _mm512_permutexvar_ps(ilo, lut_lo);
      const __m512 vhi = _mm512_permutexvar_ps(ihi, lut_hi);
      __m512 vacc = _mm512_loadu_ps(acc + v);
      vacc = _mm512_add_ps(vacc, _mm512_add_ps(vlo, vhi));
      _mm512_storeu_ps(acc + v, vacc);
    }
    for (; v < count; ++v) {
      const std::uint8_t b = codes[v];
      acc[v] += lo[b & 0x0F] + hi[b >> 4];
    }
  }
  std::memcpy(out_scores.data(), acc, count * sizeof(float));
}

/// bits=2: blocked float LUT (exact).
void score_blocked_nibble2(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                           std::span<float> out_scores) {
  score_blocked_float_lut(lut, blocked, block, out_scores);
}

/// bits=8 FastScan: AVX-512 gather from 256-entry float LUTs (exact).
void score_blocked_bits8(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                         std::span<float> out_scores) {
  const std::size_t count = blocked.block_count(block);
  const std::size_t groups = lut.num_groups();
  alignas(64) float acc[BlockedCodes::kBlock] = {};

  for (std::size_t g = 0; g < groups; ++g) {
    const std::uint8_t* codes = blocked.group_bytes_unchecked(block, g);
    const float* table = lut.table(g);
#if defined(_MSC_VER)
    if (g + 1 < groups) {
      _mm_prefetch(reinterpret_cast<const char*>(blocked.group_bytes_unchecked(block, g + 1)),
                   _MM_HINT_T0);
    }
#else
    if (g + 1 < groups) {
      __builtin_prefetch(blocked.group_bytes_unchecked(block, g + 1), 0, 3);
    }
#endif
    std::size_t v = 0;
    for (; v + 16 <= count; v += 16) {
      const __m128i bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + v));
      const __m512i idx = _mm512_cvtepu8_epi32(bytes);
      const __m512 vals = _mm512_i32gather_ps(idx, table, 4);
      const __m512 vacc = _mm512_loadu_ps(acc + v);
      _mm512_storeu_ps(acc + v, _mm512_add_ps(vacc, vals));
    }
    for (; v < count; ++v) {
      acc[v] += table[codes[v]];
    }
  }
  std::memcpy(out_scores.data(), acc, count * sizeof(float));
}

}  // namespace

void score_blocked_batch(const QueryLut& lut, const BlockedCodes& blocked, std::size_t block,
                         std::span<float> out_scores) {
  if (lut.empty() || blocked.empty()) {
    throw Error("score_blocked_batch: empty lut or blocked codes");
  }
  if (block >= blocked.n_blocks()) {
    throw Error("score_blocked_batch: block out of range");
  }
  const std::size_t count = blocked.block_count(block);
  if (out_scores.size() < count) {
    throw Error("score_blocked_batch: out_scores too small");
  }

  if (lut.has_bit1_deltas() && blocked.has_bit1_words() && lut.bits() == 1) {
    score_blocked_bit1(lut, blocked, block, out_scores);
    return;
  }

  if (lut.bits() == 4 && blocked.bits() == 4 && lut.num_groups() == blocked.num_groups()) {
    score_blocked_nibble4(lut, blocked, block, out_scores);
    return;
  }

  if (lut.bits() == 8 && blocked.bits() == 8 && lut.num_groups() == blocked.num_groups()) {
    score_blocked_bits8(lut, blocked, block, out_scores);
    return;
  }

  if (lut.bits() == 2 && blocked.bits() == 2 && lut.num_groups() == blocked.num_groups()) {
    score_blocked_nibble2(lut, blocked, block, out_scores);
    return;
  }

  if (lut.bits() == blocked.bits() && lut.num_groups() == blocked.num_groups()) {
    score_blocked_float_lut(lut, blocked, block, out_scores);
    return;
  }

  throw Error("score_blocked_batch: unsupported lut/blocked combination");
}

}  // namespace vectorcache::query
