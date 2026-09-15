#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::quantize {

namespace {

constexpr double kPi = 3.14159265358979323846;

double gaussian_pdf(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * kPi);
}

double gaussian_cdf(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

/// Conditional expectation E[X | a < X <= b] for X ~ N(0,1).
double gaussian_conditional_mean(double a, double b) {
  const double fa = std::isfinite(a) ? gaussian_pdf(a) : 0.0;
  const double fb = std::isfinite(b) ? gaussian_pdf(b) : 0.0;
  const double ca = std::isfinite(a) ? gaussian_cdf(a) : 0.0;
  const double cb = std::isfinite(b) ? gaussian_cdf(b) : 1.0;
  const double mass = cb - ca;
  if (mass < 1e-15) {
    if (std::isfinite(a) && std::isfinite(b)) {
      return 0.5 * (a + b);
    }
    return 0.0;
  }
  return (fa - fb) / mass;
}

std::vector<double> lloyd_max_std_normal(std::size_t bits) {
  const std::size_t k = std::size_t{1} << bits;
  constexpr double kRange = 4.0;
  std::vector<double> centroids(k);
  for (std::size_t i = 0; i < k; ++i) {
    centroids[i] = -kRange + (2.0 * kRange) * (static_cast<double>(i) + 0.5) / static_cast<double>(k);
  }

  std::vector<double> boundaries(k > 1 ? k - 1 : 0);
  constexpr int kMaxIters = 200;
  constexpr double kTol = 1e-10;
  for (int iter = 0; iter < kMaxIters; ++iter) {
    for (std::size_t i = 0; i + 1 < k; ++i) {
      boundaries[i] = 0.5 * (centroids[i] + centroids[i + 1]);
    }

    double max_delta = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      const double lo = (i == 0) ? -std::numeric_limits<double>::infinity() : boundaries[i - 1];
      const double hi =
          (i + 1 == k) ? std::numeric_limits<double>::infinity() : boundaries[i];
      const double updated = gaussian_conditional_mean(lo, hi);
      max_delta = std::max(max_delta, std::abs(updated - centroids[i]));
      centroids[i] = updated;
    }
    if (max_delta < kTol) {
      break;
    }
  }

  // Enforce ascending order (numerical safety).
  std::sort(centroids.begin(), centroids.end());
  return centroids;
}

}  // namespace

void validate_bits_per_dim(std::size_t bits) {
  if (bits < kMinBitsPerDim || bits > kMaxBitsPerDim) {
    throw Error("bits_per_dim must be in [" + std::to_string(kMinBitsPerDim) + ", " +
                std::to_string(kMaxBitsPerDim) + "], got " + std::to_string(bits));
  }
}

std::size_t l0_bits_per_vector(std::size_t dim, std::size_t bits) {
  validate_bits_per_dim(bits);
  return dim * bits;
}

std::size_t l0_words_per_vector(std::size_t dim, std::size_t bits) {
  const std::size_t total_bits = l0_bits_per_vector(dim, bits);
  return (total_bits + 63) / 64;
}

LloydMaxCodebook::LloydMaxCodebook(std::size_t srht_dim, std::size_t bits)
    : srht_dim_(srht_dim), bits_(bits) {
  if (srht_dim_ == 0) {
    throw Error("LloydMaxCodebook: srht_dim must be > 0");
  }
  validate_bits_per_dim(bits_);

  const auto std_centroids = lloyd_max_std_normal(bits_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(srht_dim_));
  centroids_.resize(std_centroids.size());
  for (std::size_t i = 0; i < std_centroids.size(); ++i) {
    centroids_[i] = static_cast<float>(std_centroids[i]) * scale;
  }

  boundaries_.resize(centroids_.size() > 1 ? centroids_.size() - 1 : 0);
  for (std::size_t i = 0; i + 1 < centroids_.size(); ++i) {
    boundaries_[i] = 0.5f * (centroids_[i] + centroids_[i + 1]);
  }
}

std::uint32_t LloydMaxCodebook::encode(float x) const {
  const std::size_t k = centroids_.size();
  if (k == 0) {
    throw Error("LloydMaxCodebook::encode on empty codebook");
  }
  std::uint32_t idx = 0;
  while (idx < boundaries_.size() && x > boundaries_[idx]) {
    ++idx;
  }
  return idx;
}

std::uint32_t unpack_code(std::span<const std::uint64_t> words, std::size_t dim_index,
                          std::size_t bits) {
  validate_bits_per_dim(bits);
  const std::size_t bit_pos = dim_index * bits;
  const std::size_t word_i = bit_pos / 64;
  const std::size_t bit_i = bit_pos % 64;
  const std::uint64_t mask = (bits == 64) ? ~0ull : ((std::uint64_t{1} << bits) - 1ull);
  if (bit_i + bits <= 64) {
    if (word_i >= words.size()) {
      throw Error("unpack_code: word index out of range");
    }
    return static_cast<std::uint32_t>((words[word_i] >> bit_i) & mask);
  }
  // Spans two words.
  if (word_i + 1 >= words.size()) {
    throw Error("unpack_code: straddling word index out of range");
  }
  const std::size_t lo_bits = 64 - bit_i;
  const std::uint64_t lo = words[word_i] >> bit_i;
  const std::uint64_t hi = words[word_i + 1] & ((std::uint64_t{1} << (bits - lo_bits)) - 1ull);
  return static_cast<std::uint32_t>((lo | (hi << lo_bits)) & mask);
}

std::size_t quantize_1dim_to_nbit_into(std::span<const float> vector,
                                       const LloydMaxCodebook& codebook,
                                       std::span<std::uint64_t> out) {
  const std::size_t dim = vector.size();
  const std::size_t bits = codebook.bits();
  if (dim != codebook.srht_dim()) {
    throw Error("quantize_1dim_to_nbit_into: vector dim must match codebook srht_dim");
  }
  const std::size_t need_words = l0_words_per_vector(dim, bits);
  if (out.size() < need_words) {
    throw Error("quantize_1dim_to_nbit_into: output too small");
  }
  std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(need_words), 0);

  const std::uint64_t mask = (bits == 64) ? ~0ull : ((std::uint64_t{1} << bits) - 1ull);
  for (std::size_t d = 0; d < dim; ++d) {
    const std::uint64_t code = static_cast<std::uint64_t>(codebook.encode(vector[d])) & mask;
    const std::size_t bit_pos = d * bits;
    const std::size_t word_i = bit_pos / 64;
    const std::size_t bit_i = bit_pos % 64;
    if (bit_i + bits <= 64) {
      out[word_i] |= code << bit_i;
    } else {
      const std::size_t lo_bits = 64 - bit_i;
      out[word_i] |= code << bit_i;
      out[word_i + 1] |= code >> lo_bits;
    }
  }
  return dim * bits;
}

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook) {
  const std::size_t num_words = l0_words_per_vector(vector.size(), codebook.bits());
  std::vector<std::uint64_t> words(num_words, 0);
  const std::size_t num_bits = quantize_1dim_to_nbit_into(vector, codebook, words);
  return {std::move(words), num_bits};
}

}  // namespace vectorcache::quantize
