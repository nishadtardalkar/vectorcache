#include "vectorcache/query/distance.hpp"

#include <immintrin.h>

#include <limits>

#include "vectorcache/error.hpp"

namespace vectorcache::query {

namespace {

inline std::size_t hsum_epi64(__m512i v) {
  return static_cast<std::size_t>(_mm512_reduce_add_epi64(v));
}

inline std::uint32_t hsum_epi64_256(__m256i v) {
  const __m128i lo = _mm256_castsi256_si128(v);
  const __m128i hi = _mm256_extracti128_si256(v, 1);
  const __m128i sum = _mm_add_epi64(lo, hi);
  return static_cast<std::uint32_t>(static_cast<std::uint64_t>(_mm_cvtsi128_si64(sum)) +
                                    static_cast<std::uint64_t>(_mm_extract_epi64(sum, 1)));
}

/// Cheap partial popcount sum for early-reject (avoids full reduce_add on hot path).
inline std::uint32_t partial_hsum_epi64(__m512i v) {
  const __m256i lo = _mm512_castsi512_si256(v);
  const __m256i hi = _mm512_extracti64x4_epi64(v, 1);
  return hsum_epi64_256(_mm256_add_epi64(lo, hi));
}

/// Disagree over full 64-bit words (num_bits must be a multiple of 64).
std::uint32_t disagree_full_words(std::span<const std::uint64_t> query_words,
                                  std::span<const std::uint64_t> data_words,
                                  std::size_t full_words) {
  __m512i acc = _mm512_setzero_si512();
  std::size_t i = 0;
  for (; i + 8 <= full_words; i += 8) {
    const __m512i q = _mm512_loadu_si512(query_words.data() + i);
    const __m512i d = _mm512_loadu_si512(data_words.data() + i);
    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(q, d)));
  }
  if (i < full_words) {
    const __mmask8 mask = static_cast<__mmask8>((1u << (full_words - i)) - 1u);
    const __m512i q = _mm512_maskz_loadu_epi64(mask, query_words.data() + i);
    const __m512i d = _mm512_maskz_loadu_epi64(mask, data_words.data() + i);
    acc = _mm512_mask_add_epi64(acc, mask, acc,
                                _mm512_popcnt_epi64(_mm512_xor_si512(q, d)));
  }
  return static_cast<std::uint32_t>(hsum_epi64(acc));
}

/// GloVe: 4 u64 (256 bits) per vector — two vectors per ZMM.
void batch_words4_disagree(std::span<const std::uint64_t> query_words,
                           std::span<const std::uint64_t> data_words, std::size_t num_vectors,
                           std::span<std::uint32_t> out_disagree) {
  const __m256i q256 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(query_words.data()));
  const __m512i q = _mm512_broadcast_i64x4(q256);

  std::size_t v = 0;
  for (; v + 2 <= num_vectors; v += 2) {
    const std::uint64_t* row0 = data_words.data() + v * 4;
    if (v + 6 < num_vectors) {
      _mm_prefetch(reinterpret_cast<const char*>(row0 + 24), _MM_HINT_T0);
    }
    const __m512i d = _mm512_loadu_si512(row0);
    const __m512i pc = _mm512_popcnt_epi64(_mm512_xor_si512(q, d));
    const __m256i lo = _mm512_castsi512_si256(pc);
    const __m256i hi = _mm512_extracti64x4_epi64(pc, 1);
    out_disagree[v] = hsum_epi64_256(lo);
    out_disagree[v + 1] = hsum_epi64_256(hi);
  }
  if (v < num_vectors) {
    // Remnant: stay on ZMM VPOPCNTDQ (avoids _mm256_popcnt_epi64, which needs AVX512VL).
    constexpr __mmask8 mask = 0x0f;
    const __m512i q_lo = _mm512_maskz_loadu_epi64(mask, query_words.data());
    const __m512i d = _mm512_maskz_loadu_epi64(mask, data_words.data() + v * 4);
    out_disagree[v] =
        static_cast<std::uint32_t>(hsum_epi64(_mm512_popcnt_epi64(_mm512_xor_si512(q_lo, d))));
  }
}

template <std::size_t WordsPerVec>
void batch_words_aligned_disagree(std::span<const std::uint64_t> query_words,
                                  std::span<const std::uint64_t> data_words,
                                  std::size_t num_vectors, std::span<std::uint32_t> out_disagree,
                                  std::uint32_t reject_threshold) {
  static_assert(WordsPerVec % 8 == 0);
  constexpr std::size_t kBlocks = WordsPerVec / 8;
  alignas(64) __m512i q_blocks[kBlocks];
  for (std::size_t b = 0; b < kBlocks; ++b) {
    q_blocks[b] = _mm512_load_si512(query_words.data() + b * 8);
  }

  const bool early_exit = reject_threshold != std::numeric_limits<std::uint32_t>::max();
  std::size_t v = 0;

  // 2-vector ILP: hide reduce_add latency across a pair of rows.
  for (; v + 2 <= num_vectors; v += 2) {
    const std::uint64_t* row0 = data_words.data() + v * WordsPerVec;
    const std::uint64_t* row1 = row0 + WordsPerVec;
    if (v + 6 < num_vectors) {
      _mm_prefetch(reinterpret_cast<const char*>(row0 + 4 * WordsPerVec), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(row0 + 5 * WordsPerVec), _MM_HINT_T0);
    }

    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    bool done0 = false;
    bool done1 = false;
    for (std::size_t b = 0; b < kBlocks; ++b) {
      if (!done0) {
        const __m512i d0 = _mm512_load_si512(row0 + b * 8);
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(q_blocks[b], d0)));
        // Check every 2 blocks with a cheap partial sum.
        if (early_exit && (b & 1u) == 1u && b + 1 < kBlocks) {
          const auto partial = partial_hsum_epi64(acc0);
          if (partial > reject_threshold) {
            out_disagree[v] = partial;
            done0 = true;
          }
        }
      }
      if (!done1) {
        const __m512i d1 = _mm512_load_si512(row1 + b * 8);
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(q_blocks[b], d1)));
        if (early_exit && (b & 1u) == 1u && b + 1 < kBlocks) {
          const auto partial = partial_hsum_epi64(acc1);
          if (partial > reject_threshold) {
            out_disagree[v + 1] = partial;
            done1 = true;
          }
        }
      }
      if (done0 && done1) {
        break;
      }
    }
    if (!done0) {
      out_disagree[v] = static_cast<std::uint32_t>(hsum_epi64(acc0));
    }
    if (!done1) {
      out_disagree[v + 1] = static_cast<std::uint32_t>(hsum_epi64(acc1));
    }
  }

  for (; v < num_vectors; ++v) {
    const std::uint64_t* row = data_words.data() + v * WordsPerVec;
    if (v + 4 < num_vectors) {
      _mm_prefetch(reinterpret_cast<const char*>(row + 4 * WordsPerVec), _MM_HINT_T0);
    }
    __m512i acc = _mm512_setzero_si512();
    bool rejected = false;
    for (std::size_t b = 0; b < kBlocks; ++b) {
      const __m512i d = _mm512_load_si512(row + b * 8);
      acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(q_blocks[b], d)));
      if (early_exit && (b & 1u) == 1u && b + 1 < kBlocks) {
        const auto partial = partial_hsum_epi64(acc);
        if (partial > reject_threshold) {
          out_disagree[v] = partial;
          rejected = true;
          break;
        }
      }
    }
    if (!rejected) {
      out_disagree[v] = static_cast<std::uint32_t>(hsum_epi64(acc));
    }
  }
}

/// Full-word rows where words_per_vec is a multiple of 8 (aligned ZMM blocks only).
void batch_words_zmm_disagree(std::span<const std::uint64_t> query_words, std::size_t full_words,
                              std::span<const std::uint64_t> data_words, std::size_t num_vectors,
                              std::span<std::uint32_t> out_disagree,
                              std::uint32_t reject_threshold) {
  const bool early_exit = reject_threshold != std::numeric_limits<std::uint32_t>::max();
  std::size_t v = 0;

  for (; v + 2 <= num_vectors; v += 2) {
    const std::uint64_t* row0 = data_words.data() + v * full_words;
    const std::uint64_t* row1 = row0 + full_words;
    if (v + 6 < num_vectors) {
      _mm_prefetch(reinterpret_cast<const char*>(row0 + 4 * full_words), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(row0 + 5 * full_words), _MM_HINT_T0);
    }

    __m512i acc0 = _mm512_setzero_si512();
    __m512i acc1 = _mm512_setzero_si512();
    bool done0 = false;
    bool done1 = false;
    std::size_t block = 0;
    for (std::size_t i = 0; i < full_words; i += 8, ++block) {
      const __m512i q = _mm512_loadu_si512(query_words.data() + i);
      if (!done0) {
        const __m512i d0 = _mm512_load_si512(row0 + i);
        acc0 = _mm512_add_epi64(acc0, _mm512_popcnt_epi64(_mm512_xor_si512(q, d0)));
        if (early_exit && (block & 1u) == 1u && i + 8 < full_words) {
          const auto partial = partial_hsum_epi64(acc0);
          if (partial > reject_threshold) {
            out_disagree[v] = partial;
            done0 = true;
          }
        }
      }
      if (!done1) {
        const __m512i d1 = _mm512_load_si512(row1 + i);
        acc1 = _mm512_add_epi64(acc1, _mm512_popcnt_epi64(_mm512_xor_si512(q, d1)));
        if (early_exit && (block & 1u) == 1u && i + 8 < full_words) {
          const auto partial = partial_hsum_epi64(acc1);
          if (partial > reject_threshold) {
            out_disagree[v + 1] = partial;
            done1 = true;
          }
        }
      }
      if (done0 && done1) {
        break;
      }
    }
    if (!done0) {
      out_disagree[v] = static_cast<std::uint32_t>(hsum_epi64(acc0));
    }
    if (!done1) {
      out_disagree[v + 1] = static_cast<std::uint32_t>(hsum_epi64(acc1));
    }
  }

  for (; v < num_vectors; ++v) {
    const std::uint64_t* row = data_words.data() + v * full_words;
    if (v + 4 < num_vectors) {
      _mm_prefetch(reinterpret_cast<const char*>(row + 4 * full_words), _MM_HINT_T0);
    }
    __m512i acc = _mm512_setzero_si512();
    bool rejected = false;
    std::size_t block = 0;
    for (std::size_t i = 0; i < full_words; i += 8, ++block) {
      const __m512i q = _mm512_loadu_si512(query_words.data() + i);
      const __m512i d = _mm512_load_si512(row + i);
      acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_xor_si512(q, d)));
      if (early_exit && (block & 1u) == 1u && i + 8 < full_words) {
        const auto partial = partial_hsum_epi64(acc);
        if (partial > reject_threshold) {
          out_disagree[v] = partial;
          rejected = true;
          break;
        }
      }
    }
    if (!rejected) {
      out_disagree[v] = static_cast<std::uint32_t>(hsum_epi64(acc));
    }
  }
}

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
  if (num_bits % 64 != 0 || query_words.size() < num_bits / 64) {
    throw Error("bit_agreement_score requires num_bits multiple of 64");
  }
  const std::size_t full_words = num_bits / 64;
  const std::uint32_t disagree = disagree_full_words(query_words, data_words, full_words);
  return agreement_from_agree(num_bits - disagree, num_bits);
}

void bit_agreement_batch_disagree(std::span<const std::uint64_t> query_words,
                                  std::size_t num_bits, std::span<const std::uint64_t> data_words,
                                  std::size_t data_words_per_vec, std::size_t num_vectors,
                                  std::span<std::uint32_t> out_disagree,
                                  std::uint32_t reject_threshold) {
  if (out_disagree.size() < num_vectors || num_vectors == 0 || num_bits == 0) {
    return;
  }
  if (query_words.size() < data_words_per_vec) {
    throw Error("bit_agreement_batch_disagree: query words too short");
  }
  if (data_words.size() < num_vectors * data_words_per_vec) {
    throw Error("bit_agreement_batch_disagree: data words too short");
  }
  if (num_bits % 64 != 0 || num_bits / 64 != data_words_per_vec) {
    throw Error("bit_agreement_batch_disagree requires full-word L0 (num_bits == 64 * words)");
  }

  const std::size_t full_words = data_words_per_vec;
  if (full_words == 4) {
    batch_words4_disagree(query_words, data_words, num_vectors, out_disagree);
    return;
  }
  if (full_words == 32) {
    batch_words_aligned_disagree<32>(query_words, data_words, num_vectors, out_disagree,
                                     reject_threshold);
    return;
  }
  if (full_words == 64) {
    batch_words_aligned_disagree<64>(query_words, data_words, num_vectors, out_disagree,
                                     reject_threshold);
    return;
  }
  if ((full_words % 8) == 0) {
    batch_words_zmm_disagree(query_words, full_words, data_words, num_vectors, out_disagree,
                             reject_threshold);
    return;
  }
  // Non-multiple-of-8 word counts (e.g. unit-test dims): masked ZMM path per vector.
  for (std::size_t v = 0; v < num_vectors; ++v) {
    const std::size_t offset = v * data_words_per_vec;
    out_disagree[v] = disagree_full_words(query_words.subspan(0, data_words_per_vec),
                                          data_words.subspan(offset, data_words_per_vec),
                                          full_words);
  }
}

void bit_agreement_batch(std::span<const std::uint64_t> query_words, std::size_t num_bits,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, std::span<float> out_scores) {
  if (out_scores.size() < num_vectors || num_vectors == 0 || num_bits == 0) {
    return;
  }
  constexpr std::size_t kMaxBatch = 64;
  if (num_vectors > kMaxBatch) {
    throw Error("bit_agreement_batch: num_vectors exceeds 64 (use bit_agreement_batch_disagree)");
  }
  alignas(64) std::uint32_t disagree[kMaxBatch];
  bit_agreement_batch_disagree(query_words, num_bits, data_words, data_words_per_vec, num_vectors,
                               std::span<std::uint32_t>(disagree, num_vectors));
  for (std::size_t v = 0; v < num_vectors; ++v) {
    out_scores[v] = score_from_disagree(disagree[v], num_bits);
  }
}

}  // namespace vectorcache::query
