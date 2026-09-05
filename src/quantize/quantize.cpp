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

}  // namespace

std::size_t l0_bits_per_vector(std::size_t dim) { return dim; }

std::size_t l0_words_per_vector(std::size_t dim) { return (dim + 63) / 64; }

std::uint8_t quantize_parent_8bit(std::span<const float> vector) {
  const std::size_t dim = vector.size();
  if (dim == 0 || dim % PARENT_BITS != 0) {
    throw Error("quantize_parent_8bit requires dim > 0 and dim % 8 == 0");
  }

  const std::size_t chunk = dim / PARENT_BITS;
  std::uint8_t key = 0;

  for (std::size_t bit = 0; bit < PARENT_BITS; ++bit) {
    const float* base = vector.data() + bit * chunk;
    float sum = 0.0f;
    std::size_t d = 0;
    while (d + simd::kWidth <= chunk) {
      const __m512 values = load_vector_block(base + d);
      sum += _mm512_reduce_add_ps(values);
      d += simd::kWidth;
    }
    while (d < chunk) {
      sum += base[d];
      ++d;
    }
    if (sum >= 0.0f) {
      key = static_cast<std::uint8_t>(key | static_cast<std::uint8_t>(1u << bit));
    }
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
