#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

#include "vectorcache/simd.hpp"

namespace vectorcache::quantize {

namespace {

constexpr std::size_t kBlocksPerZmm = 4;
constexpr std::size_t kBlocksPerWord = 64;

const __m512 kHadamardPatterns = _mm512_setr_ps(
    1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f);

std::uint32_t project_4blocks_sign_bits(__m512 values) {
  const __m512 prod = _mm512_mul_ps(values, kHadamardPatterns);
  const __m512 hadd = _mm512_add_ps(prod, _mm512_movehdup_ps(prod));
  const __m512 sums = _mm512_add_ps(hadd, _mm512_permute_ps(hadd, 0x4E));
  const __mmask16 ge = _mm512_cmp_ps_mask(sums, _mm512_setzero_ps(), _CMP_GE_OQ);
  return static_cast<std::uint32_t>((ge >> 0) & 1u) | static_cast<std::uint32_t>(((ge >> 4) & 1u) << 1) |
         static_cast<std::uint32_t>(((ge >> 8) & 1u) << 2) | static_cast<std::uint32_t>(((ge >> 12) & 1u) << 3);
}

std::uint32_t project_partial_blocks_sign_bits(std::span<const float> vector, std::size_t block,
                                               std::size_t block_count) {
  alignas(64) float buf[simd::kWidth] = {};
  for (std::size_t b = 0; b < block_count; ++b) {
    const std::size_t offset = (block + b) * 4;
    const std::size_t copy_len = std::min<std::size_t>(4, vector.size() - offset);
    std::memcpy(buf + b * 4, vector.data() + offset, copy_len * sizeof(float));
  }
  return project_4blocks_sign_bits(_mm512_load_ps(buf));
}

__m512 load_vector_block(const float* ptr) {
  if ((reinterpret_cast<std::uintptr_t>(ptr) & 0x3F) == 0) {
    return _mm512_load_ps(ptr);
  }
  return _mm512_loadu_ps(ptr);
}

}  // namespace

std::size_t l1_bits_per_vector(std::size_t dim) { return (dim + 3) / 4; }

std::size_t l1_words_per_vector(std::size_t dim) {
  const std::size_t num_bits = l1_bits_per_vector(dim);
  return (num_bits + 63) / 64;
}

std::size_t quantize_4d_to_1bit_into(std::span<const float> vector,
                                     std::span<std::uint64_t> out) {
  const std::size_t dim = vector.size();
  const std::size_t num_4d_blocks = l1_bits_per_vector(dim);
  const bool aligned_words = (num_4d_blocks % kBlocksPerWord) == 0;
  if (!aligned_words) {
    std::fill(out.begin(), out.end(), 0);
  }

  std::size_t b = 0;
  while (b + kBlocksPerWord <= num_4d_blocks) {
    std::uint64_t word = 0;
    for (std::size_t chunk = 0; chunk < kBlocksPerWord / kBlocksPerZmm; ++chunk) {
      const std::size_t block = b + chunk * kBlocksPerZmm;
      const __m512 values = load_vector_block(vector.data() + block * 4);
      word |= static_cast<std::uint64_t>(project_4blocks_sign_bits(values)) << (chunk * kBlocksPerZmm);
    }
    out[b / kBlocksPerWord] = word;
    b += kBlocksPerWord;
  }

  if (b < num_4d_blocks) {
    std::uint64_t word = 0;
    const std::size_t remaining = num_4d_blocks - b;
    std::size_t bit = 0;
    while (bit + kBlocksPerZmm <= remaining) {
      const std::size_t block = b + bit;
      const __m512 values = load_vector_block(vector.data() + block * 4);
      word |= static_cast<std::uint64_t>(project_4blocks_sign_bits(values)) << bit;
      bit += kBlocksPerZmm;
    }
    if (bit < remaining) {
      const std::size_t tail_blocks = remaining - bit;
      const std::uint32_t tail_bits = project_partial_blocks_sign_bits(vector, b + bit, tail_blocks) &
                                      ((1u << tail_blocks) - 1u);
      word |= static_cast<std::uint64_t>(tail_bits) << bit;
    }
    out[b / kBlocksPerWord] = word;
  }

  return num_4d_blocks;
}

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_4d_to_1bit(
    std::span<const float> vector) {
  const std::size_t num_words = l1_words_per_vector(vector.size());
  std::vector<std::uint64_t> words(num_words, 0);
  const std::size_t num_bits = quantize_4d_to_1bit_into(vector, words);
  return {std::move(words), num_bits};
}

}  // namespace vectorcache::quantize
