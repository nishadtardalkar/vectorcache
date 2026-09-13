#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::quantize {

namespace {

__m512 load_vector_block(const float* ptr) {
  if ((reinterpret_cast<std::uintptr_t>(ptr) & 0x3F) == 0) {
    return _mm512_load_ps(ptr);
  }
  return _mm512_loadu_ps(ptr);
}

void validate_parent_dim(std::size_t dim) {
  if (dim == 0 || dim % PARENT_GROUPS != 0) {
    throw Error("parent quantize requires dim > 0 and dim % 8 == 0");
  }
  const std::size_t chunk = dim / PARENT_GROUPS;
  if (chunk < 2) {
    throw Error("parent quantize requires chunk size >= 2 (dim >= 16 with 8 groups)");
  }
}

float sum_range(const float* base, std::size_t n) {
  float sum = 0.0f;
  std::size_t d = 0;
  while (d + simd::kWidth <= n) {
    sum += _mm512_reduce_add_ps(load_vector_block(base + d));
    d += simd::kWidth;
  }
  while (d < n) {
    sum += base[d];
    ++d;
  }
  return sum;
}

/// s1 = sum(chunk) = <x, all-ones>; s2 = x[0]-x[1] = <x, (1,-1,0,...)>.
void chunk_project_2d(const float* base, std::size_t chunk, float& s1, float& s2) {
  s1 = sum_range(base, chunk);
  s2 = base[0] - base[1];
}

}  // namespace

std::size_t l0_bits_per_vector(std::size_t dim) { return dim; }

std::size_t l0_words_per_vector(std::size_t dim) { return (dim + 63) / 64; }

ParentFoldBits quantize_parent_fold_bits(std::span<const float> vector) {
  validate_parent_dim(vector.size());
  const std::size_t chunk = vector.size() / PARENT_GROUPS;
  ParentFoldBits folds{};
  for (std::size_t g = 0; g < PARENT_GROUPS; ++g) {
    float s1 = 0.0f;
    float s2 = 0.0f;
    chunk_project_2d(vector.data() + g * chunk, chunk, s1, s2);
    if (s1 >= 0.0f) {
      folds.b1 = static_cast<std::uint8_t>(folds.b1 | static_cast<std::uint8_t>(1u << g));
    }
    if (s2 >= 0.0f) {
      folds.b2 = static_cast<std::uint8_t>(folds.b2 | static_cast<std::uint8_t>(1u << g));
    }
  }
  return folds;
}

std::array<std::uint16_t, PARENT_POSTINGS> parent_posting_keys(ParentFoldBits folds) {
  std::array<std::uint16_t, PARENT_POSTINGS> keys{};
  // Per group: two allowed nibbles encode(F1,b1) and encode(F2,b2).
  std::uint16_t nibble_f1[PARENT_GROUPS];
  std::uint16_t nibble_f2[PARENT_GROUPS];
  for (std::size_t g = 0; g < PARENT_GROUPS; ++g) {
    const std::uint8_t bit1 = static_cast<std::uint8_t>((folds.b1 >> g) & 1u);
    const std::uint8_t bit2 = static_cast<std::uint8_t>((folds.b2 >> g) & 1u);
    nibble_f1[g] = parent_nibble(0, bit1);
    nibble_f2[g] = parent_nibble(1, bit2);
  }

  for (std::size_t mask = 0; mask < PARENT_POSTINGS; ++mask) {
    std::uint16_t key = 0;
    for (std::size_t g = 0; g < PARENT_GROUPS; ++g) {
      const std::uint16_t nibble = ((mask >> g) & 1u) != 0u ? nibble_f2[g] : nibble_f1[g];
      key = static_cast<std::uint16_t>(key | parent_pack_nibble(g, nibble));
    }
    keys[mask] = key;
  }
  return keys;
}

std::array<std::uint16_t, PARENT_POSTINGS> parent_posting_keys(std::span<const float> vector) {
  return parent_posting_keys(quantize_parent_fold_bits(vector));
}

std::uint16_t quantize_parent_query_key(std::span<const float> vector) {
  validate_parent_dim(vector.size());
  const std::size_t chunk = vector.size() / PARENT_GROUPS;
  std::uint16_t key = 0;
  for (std::size_t g = 0; g < PARENT_GROUPS; ++g) {
    float s1 = 0.0f;
    float s2 = 0.0f;
    chunk_project_2d(vector.data() + g * chunk, chunk, s1, s2);
    // |s1|/√C >= |s2|/√2  iff  2 s1² >= C s2²
    const bool use_f1 =
        (2.0f * s1 * s1) >= (static_cast<float>(chunk) * s2 * s2);
    const std::uint8_t fold = use_f1 ? 0u : 1u;
    const std::uint8_t bit = use_f1 ? (s1 >= 0.0f ? 1u : 0u) : (s2 >= 0.0f ? 1u : 0u);
    key = static_cast<std::uint16_t>(key | parent_pack_nibble(g, parent_nibble(fold, bit)));
  }
  return key;
}

std::size_t quantize_1dim_to_1bit_into(std::span<const float> vector,
                                       std::span<std::uint64_t> out) {
  const std::size_t dim = vector.size();
  const std::size_t need_words = l0_words_per_vector(dim);
  if (out.size() < need_words) {
    throw Error("quantize_1dim_to_1bit_into: output too small");
  }
  // Power-of-two dims write every word fully; skip zero-fill.
  if (dim % 64 != 0) {
    std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(need_words), 0);
  }

  std::size_t d = 0;
  while (d + 64 <= dim) {
    std::uint64_t word = 0;
    for (std::size_t chunk = 0; chunk < 4; ++chunk) {
      const __m512 values = load_vector_block(vector.data() + d + chunk * 16);
      const __mmask16 ge = _mm512_cmp_ps_mask(values, _mm512_setzero_ps(), _CMP_GE_OQ);
      word |= static_cast<std::uint64_t>(ge) << (chunk * 16);
    }
    out[d / 64] = word;
    d += 64;
  }

  if (d < dim) {
    std::uint64_t word = 0;
    const std::size_t remaining = dim - d;
    std::size_t bit = 0;
    while (bit + 16 <= remaining) {
      const __m512 values = load_vector_block(vector.data() + d + bit);
      const __mmask16 ge = _mm512_cmp_ps_mask(values, _mm512_setzero_ps(), _CMP_GE_OQ);
      word |= static_cast<std::uint64_t>(ge) << bit;
      bit += 16;
    }
    if (bit < remaining) {
      alignas(64) float buf[simd::kWidth] = {};
      const std::size_t tail = remaining - bit;
      std::memcpy(buf, vector.data() + d + bit, tail * sizeof(float));
      const __mmask16 ge = _mm512_cmp_ps_mask(_mm512_load_ps(buf), _mm512_setzero_ps(), _CMP_GE_OQ);
      const std::uint32_t tail_bits = static_cast<std::uint32_t>(ge) & ((1u << tail) - 1u);
      word |= static_cast<std::uint64_t>(tail_bits) << bit;
    }
    out[d / 64] = word;
  }

  return dim;
}

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_1bit(
    std::span<const float> vector) {
  const std::size_t num_words = l0_words_per_vector(vector.size());
  std::vector<std::uint64_t> words(num_words, 0);
  const std::size_t num_bits = quantize_1dim_to_1bit_into(vector, words);
  return {std::move(words), num_bits};
}

}  // namespace vectorcache::quantize
