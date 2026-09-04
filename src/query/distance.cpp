#include "vectorcache/query/distance.hpp"

#include <bit>
#include <cstring>
#include <immintrin.h>

namespace vectorcache::query {

namespace {

#if defined(__AVX512VPOPCNTDQ__) || (defined(__AVX512F__) && defined(__GNUC__))
#define VECTORCACHE_HAS_VPOPCNTDQ 1
#endif

std::size_t popcount_xor_u64(std::uint64_t xor_bits) {
#if defined(VECTORCACHE_HAS_VPOPCNTDQ)
  return static_cast<std::size_t>(_mm_popcnt_u64(xor_bits));
#else
  return static_cast<std::size_t>(std::popcount(xor_bits));
#endif
}

std::size_t agree_words_scalar(std::span<const std::uint64_t> query_words,
                               std::span<const std::uint64_t> data_words, std::size_t num_bits) {
  std::size_t agree = 0;
  std::size_t remaining = num_bits;
  for (std::size_t i = 0; i < query_words.size() && remaining > 0; ++i) {
    const std::size_t chunk = remaining < 64 ? remaining : 64;
    const std::uint64_t mask = chunk == 64 ? ~0ULL : ((1ULL << chunk) - 1ULL);
    const std::uint64_t xor_bits = (query_words[i] ^ data_words[i]) & mask;
    agree += chunk - popcount_xor_u64(xor_bits);
    remaining -= chunk;
  }
  return agree;
}

#if defined(VECTORCACHE_HAS_VPOPCNTDQ)

std::size_t agree_full_words_simd(std::span<const std::uint64_t> query_words,
                                  std::span<const std::uint64_t> data_words,
                                  std::size_t full_words) {
  std::size_t agree = 0;
  std::size_t i = 0;
  for (; i + 8 <= full_words; i += 8) {
    const __m512i q = _mm512_loadu_si512(query_words.data() + i);
    const __m512i d = _mm512_loadu_si512(data_words.data() + i);
    const __m512i x = _mm512_xor_si512(q, d);
    agree += static_cast<std::size_t>(_mm512_reduce_add_epi64(_mm512_popcnt_epi64(x)));
  }
  for (; i < full_words; ++i) {
    agree += 64 - popcount_xor_u64(query_words[i] ^ data_words[i]);
  }
  return agree;
}

#endif

float agreement_from_agree(std::size_t agree, std::size_t num_bits) {
  return (2.0f * static_cast<float>(agree) - static_cast<float>(num_bits)) /
         static_cast<float>(num_bits);
}

}  // namespace

float bit_agreement_score(std::span<const std::uint64_t> query_words,
                          std::span<const std::uint64_t> data_words, std::size_t num_bits) {
  if (query_words.size() != data_words.size() || num_bits == 0) {
    return 0.0f;
  }

  const std::size_t full_words = num_bits / 64;
  const std::size_t tail_bits = num_bits % 64;

#if defined(VECTORCACHE_HAS_VPOPCNTDQ)
  if (tail_bits == 0 && full_words > 0) {
    const std::size_t agree = agree_full_words_simd(query_words, data_words, full_words);
    return agreement_from_agree(agree, num_bits);
  }
#endif

  return agreement_from_agree(agree_words_scalar(query_words, data_words, num_bits), num_bits);
}

void bit_agreement_batch(std::span<const std::uint64_t> query_words, std::size_t num_bits,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, std::span<float> out_scores) {
  if (out_scores.size() < num_vectors) {
    return;
  }
  for (std::size_t v = 0; v < num_vectors; ++v) {
    const std::size_t offset = v * data_words_per_vec;
    out_scores[v] = bit_agreement_score(
        query_words, data_words.subspan(offset, data_words_per_vec), num_bits);
  }
}

}  // namespace vectorcache::query
