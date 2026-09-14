#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

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

std::uint8_t support_hd(const SupportKey& a, const SupportKey& b) {
  if (a.d != b.d) {
    throw Error("support_hd requires equal depth keys");
  }
  std::size_t i = 0;
  std::size_t j = 0;
  std::size_t inter = 0;
  while (i < a.d && j < b.d) {
    if (a.dims[i] == b.dims[j]) {
      ++inter;
      ++i;
      ++j;
    } else if (a.dims[i] < b.dims[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return static_cast<std::uint8_t>(2u * (static_cast<std::size_t>(a.d) - inter));
}

SupportKey quantize_support_key(std::span<const float> vector, std::size_t d) {
  if (d == 0 || d > kMaxSupportDepth) {
    throw Error("support depth must be in 1..kMaxSupportDepth");
  }
  if (vector.size() < d) {
    throw Error("vector shorter than support depth");
  }
  if (vector.size() > std::numeric_limits<std::uint16_t>::max() + 1ull) {
    throw Error("vector dim exceeds uint16 index range");
  }

  const std::size_t n = vector.size();
  // Min-heap of size d over the best candidates (root = worst of the best).
  // Tie-break: higher index wins when |x| equal → worse has smaller idx.
  struct Cand {
    float mag;
    std::uint16_t idx;
  };
  Cand best[kMaxSupportDepth];
  std::size_t filled = 0;

  auto worse = [](const Cand& a, const Cand& b) {
    return a.mag < b.mag || (a.mag == b.mag && a.idx < b.idx);
  };
  auto sift_up = [&](std::size_t i) {
    while (i > 0) {
      const std::size_t parent = (i - 1) / 2;
      if (!worse(best[i], best[parent])) {
        break;
      }
      std::swap(best[i], best[parent]);
      i = parent;
    }
  };
  auto sift_down = [&](std::size_t i) {
    while (true) {
      const std::size_t left = 2 * i + 1;
      const std::size_t right = left + 1;
      std::size_t w = i;
      if (left < filled && worse(best[left], best[w])) {
        w = left;
      }
      if (right < filled && worse(best[right], best[w])) {
        w = right;
      }
      if (w == i) {
        break;
      }
      std::swap(best[i], best[w]);
      i = w;
    }
  };

  auto consider = [&](float mag, std::uint16_t idx) {
    const Cand c{mag, idx};
    if (filled < d) {
      best[filled] = c;
      sift_up(filled);
      ++filled;
      return;
    }
    if (worse(best[0], c)) {
      // c is better than current worst-of-best.
      best[0] = c;
      sift_down(0);
    }
  };

  std::size_t i = 0;
  while (i + simd::kWidth <= n) {
    const __m512 vals = load_vector_block(vector.data() + i);
    const __m512 absv = _mm512_abs_ps(vals);
    alignas(64) float mags[simd::kWidth];
    _mm512_store_ps(mags, absv);
    for (std::size_t lane = 0; lane < simd::kWidth; ++lane) {
      consider(mags[lane], static_cast<std::uint16_t>(i + lane));
    }
    i += simd::kWidth;
  }
  while (i < n) {
    consider(std::fabs(vector[i]), static_cast<std::uint16_t>(i));
    ++i;
  }

  SupportKey key{};
  key.d = static_cast<std::uint8_t>(d);
  for (std::size_t k = 0; k < d; ++k) {
    key.dims[k] = best[k].idx;
  }
  std::sort(key.dims, key.dims + d);
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
