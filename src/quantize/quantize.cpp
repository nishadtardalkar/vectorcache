#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::quantize {

namespace {

/// Continued fraction for incomplete beta (Numerical Recipes / Cephes style).
double betacf(double a, double b, double x) {
  constexpr int kMaxIt = 200;
  constexpr double kEps = 3e-14;
  constexpr double kFpmin = 1e-300;

  const double qab = a + b;
  const double qap = a + 1.0;
  const double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::abs(d) < kFpmin) {
    d = kFpmin;
  }
  d = 1.0 / d;
  double h = d;

  for (int m = 1; m <= kMaxIt; ++m) {
    const int m2 = 2 * m;
    double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x /
                ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    h *= d * c;

    aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x /
         ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::abs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::abs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::abs(del - 1.0) < kEps) {
      break;
    }
  }
  return h;
}

/// Regularized incomplete beta I_x(a, b).
double betai(double a, double b, double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= 1.0) {
    return 1.0;
  }
  const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
  const double bt = std::exp(a * std::log(x) + b * std::log(1.0 - x) - lbeta);
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return bt * betacf(a, b, x) / a;
  }
  return 1.0 - bt * betacf(b, a, 1.0 - x) / b;
}

struct BetaAA {
  double a;
  double log_norm;  // -log B(a,a) = lgamma(2a) - 2*lgamma(a)

  explicit BetaAA(double shape) : a(shape), log_norm(std::lgamma(2.0 * shape) - 2.0 * std::lgamma(shape)) {}

  double pdf01(double t) const {
    if (t <= 0.0 || t >= 1.0) {
      return 0.0;
    }
    return std::exp((a - 1.0) * std::log(t) + (a - 1.0) * std::log(1.0 - t) + log_norm);
  }

  double cdf01(double t) const {
    if (t <= 0.0) {
      return 0.0;
    }
    if (t >= 1.0) {
      return 1.0;
    }
    return betai(a, a, t);
  }

  /// PDF of shifted Beta on [-1, 1]: X = 2T - 1.
  double pdf_shifted(double x) const { return pdf01((x + 1.0) / 2.0) / 2.0; }

  double cdf_shifted(double x) const { return cdf01((x + 1.0) / 2.0); }
};

double adaptive_simpson(const auto& f, double a, double b, double tol, int max_depth) {
  const auto rec = [&](auto&& self, double lo, double hi, double flo, double fhi, double fmid,
                       double whole, double eps, int depth) -> double {
    const double mid = 0.5 * (lo + hi);
    const double m1 = 0.5 * (lo + mid);
    const double m2 = 0.5 * (mid + hi);
    const double fm1 = f(m1);
    const double fm2 = f(m2);
    const double left = (mid - lo) / 6.0 * (flo + 4.0 * fm1 + fmid);
    const double right = (hi - mid) / 6.0 * (fmid + 4.0 * fm2 + fhi);
    const double refined = left + right;
    if (depth == 0 || std::abs(refined - whole) < 15.0 * eps) {
      return refined + (refined - whole) / 15.0;
    }
    return self(self, lo, mid, flo, fmid, fm1, left, eps * 0.5, depth - 1) +
           self(self, mid, hi, fmid, fhi, fm2, right, eps * 0.5, depth - 1);
  };

  const double mid = 0.5 * (a + b);
  const double fa = f(a);
  const double fb = f(b);
  const double fm = f(mid);
  const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  return rec(rec, a, b, fa, fb, fm, whole, tol, max_depth);
}

/// Inverse CDF of Beta(a,a) on [0,1] via bisection, then map to [-1,1].
double sample_shifted_beta(const BetaAA& beta, double u) {
  u = std::clamp(u, 1e-12, 1.0 - 1e-12);
  double lo = 0.0;
  double hi = 1.0;
  for (int i = 0; i < 64; ++i) {
    const double mid = 0.5 * (lo + hi);
    if (beta.cdf01(mid) < u) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return 2.0 * (0.5 * (lo + hi)) - 1.0;
}

/// Lloyd-Max for Beta((dim-1)/2,(dim-1)/2) on [-1, 1] (TurboVec codebook.rs).
std::pair<std::vector<float>, std::vector<float>> lloyd_max_beta(std::size_t bits, std::size_t dim) {
  const double a = (static_cast<double>(dim) - 1.0) / 2.0;
  const BetaAA beta(a);
  const std::size_t n_levels = std::size_t{1} << bits;

  // std of Beta on [-1,1]: sqrt(2a / ((2a+1) * 4a))
  const double std_dev = std::sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a));
  const double spread = 3.0 * std_dev;
  std::vector<double> centroids(n_levels);
  if (n_levels == 1) {
    centroids[0] = 0.0;
  } else {
    for (std::size_t i = 0; i < n_levels; ++i) {
      centroids[i] = -spread + 2.0 * spread * static_cast<double>(i) / static_cast<double>(n_levels - 1);
    }
  }

  constexpr int kMaxIters = 200;
  constexpr double kTol = 1e-12;
  for (int iter = 0; iter < kMaxIters; ++iter) {
    std::vector<double> boundaries(n_levels > 1 ? n_levels - 1 : 0);
    for (std::size_t i = 0; i + 1 < n_levels; ++i) {
      boundaries[i] = 0.5 * (centroids[i] + centroids[i + 1]);
    }

    std::vector<double> edges;
    edges.reserve(n_levels + 1);
    edges.push_back(-1.0);
    edges.insert(edges.end(), boundaries.begin(), boundaries.end());
    edges.push_back(1.0);

    std::vector<double> new_centroids(n_levels);
    for (std::size_t i = 0; i < n_levels; ++i) {
      const double lo = edges[i];
      const double hi = edges[i + 1];
      const double cdf_lo = beta.cdf_shifted(lo);
      const double cdf_hi = beta.cdf_shifted(hi);
      const double prob = cdf_hi - cdf_lo;
      if (prob < 1e-15) {
        new_centroids[i] = centroids[i];
      } else {
        const double mean = adaptive_simpson(
            [&](double x) { return x * beta.pdf_shifted(x); }, lo, hi, 1e-14, 50);
        new_centroids[i] = mean / prob;
      }
    }

    double max_change = 0.0;
    for (std::size_t i = 0; i < n_levels; ++i) {
      max_change = std::max(max_change, std::abs(new_centroids[i] - centroids[i]));
    }
    centroids = std::move(new_centroids);
    if (max_change < kTol) {
      break;
    }
  }

  // Cast to f32 first, then midpoints in f32 (TurboVec cross-platform stability).
  std::vector<float> centroids_f32(n_levels);
  for (std::size_t i = 0; i < n_levels; ++i) {
    centroids_f32[i] = static_cast<float>(centroids[i]);
  }
  std::vector<float> boundaries_f32(n_levels > 1 ? n_levels - 1 : 0);
  for (std::size_t i = 0; i + 1 < n_levels; ++i) {
    boundaries_f32[i] = 0.5f * (centroids_f32[i] + centroids_f32[i + 1]);
  }
  return {std::move(boundaries_f32), std::move(centroids_f32)};
}

float squared_dist(std::span<const float> a, std::span<const float> b) {
  float s = 0.0f;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

std::uint32_t nearest_centroid(std::span<const float> x, std::span<const float> centroids,
                               std::size_t k, std::size_t d) {
  std::uint32_t best = 0;
  float best_dist = std::numeric_limits<float>::infinity();
  for (std::size_t c = 0; c < k; ++c) {
    const float dist = squared_dist(x, centroids.subspan(c * d, d));
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<std::uint32_t>(c);
    }
  }
  return best;
}

/// Monte Carlo Lloyd-Max under product of shifted Beta((dim-1)/2,(dim-1)/2) on [-1,1]^d.
std::vector<float> lloyd_max_beta_block(std::size_t bits, std::size_t dim, std::size_t block_dims) {
  const double a = (static_cast<double>(dim) - 1.0) / 2.0;
  const BetaAA beta(a);
  const std::size_t k = std::size_t{1} << bits;
  const std::size_t d = block_dims;

  constexpr std::size_t kNumSamples = 131072;
  constexpr int kMaxIters = 100;
  constexpr double kTol = 1e-6;

  std::mt19937_64 rng(0xC0FFEEULL ^ (static_cast<std::uint64_t>(dim) * 0x9E3779B97F4A7C15ULL) ^
                      (static_cast<std::uint64_t>(block_dims) << 32) ^ bits);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  std::vector<float> samples(kNumSamples * d);
  for (std::size_t i = 0; i < kNumSamples; ++i) {
    for (std::size_t j = 0; j < d; ++j) {
      samples[i * d + j] = static_cast<float>(sample_shifted_beta(beta, uni(rng)));
    }
  }

  // Init: pick K distinct samples (or first K if collision).
  std::vector<float> centroids(k * d);
  {
    std::uniform_int_distribution<std::size_t> pick(0, kNumSamples - 1);
    std::vector<std::size_t> used;
    used.reserve(k);
    for (std::size_t c = 0; c < k; ++c) {
      std::size_t idx = pick(rng);
      for (int attempt = 0; attempt < 16; ++attempt) {
        if (std::find(used.begin(), used.end(), idx) == used.end()) {
          break;
        }
        idx = pick(rng);
      }
      used.push_back(idx);
      for (std::size_t j = 0; j < d; ++j) {
        centroids[c * d + j] = samples[idx * d + j];
      }
    }
  }

  std::vector<double> sums(k * d);
  std::vector<std::size_t> counts(k);
  for (int iter = 0; iter < kMaxIters; ++iter) {
    std::fill(sums.begin(), sums.end(), 0.0);
    std::fill(counts.begin(), counts.end(), 0);

    for (std::size_t i = 0; i < kNumSamples; ++i) {
      const auto x = std::span<const float>(samples.data() + i * d, d);
      const std::uint32_t c = nearest_centroid(x, centroids, k, d);
      counts[c] += 1;
      for (std::size_t j = 0; j < d; ++j) {
        sums[c * d + j] += static_cast<double>(x[j]);
      }
    }

    double max_change = 0.0;
    for (std::size_t c = 0; c < k; ++c) {
      if (counts[c] == 0) {
        // Re-seed empty cluster from a random sample.
        std::uniform_int_distribution<std::size_t> pick(0, kNumSamples - 1);
        const std::size_t idx = pick(rng);
        for (std::size_t j = 0; j < d; ++j) {
          const float neu = samples[idx * d + j];
          max_change = std::max(max_change, static_cast<double>(std::abs(neu - centroids[c * d + j])));
          centroids[c * d + j] = neu;
        }
        continue;
      }
      for (std::size_t j = 0; j < d; ++j) {
        const float neu = static_cast<float>(sums[c * d + j] / static_cast<double>(counts[c]));
        max_change = std::max(max_change, static_cast<double>(std::abs(neu - centroids[c * d + j])));
        centroids[c * d + j] = neu;
      }
    }
    if (max_change < kTol) {
      break;
    }
  }

  return centroids;
}

void pack_code(std::span<std::uint64_t> out, std::size_t bit_pos, std::size_t bits,
               std::uint64_t code) {
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

}  // namespace

void validate_bits_per_dim(std::size_t bits) {
  if (bits < kMinBitsPerDim || bits > kMaxBitsPerDim) {
    throw Error("bits_per_dim must be in [" + std::to_string(kMinBitsPerDim) + ", " +
                std::to_string(kMaxBitsPerDim) + "], got " + std::to_string(bits));
  }
}

void validate_block_dims(std::size_t block_dims) {
  if (block_dims < kMinBlockDims || block_dims > kMaxBlockDims) {
    throw Error("block_dims must be in [" + std::to_string(kMinBlockDims) + ", " +
                std::to_string(kMaxBlockDims) + "], got " + std::to_string(block_dims));
  }
}

void validate_quant_spec(std::size_t dim, QuantSpec spec) {
  validate_bits_per_dim(spec.bits);
  validate_block_dims(spec.block_dims);
  if (dim < 2) {
    throw Error("quantize: dim must be >= 2");
  }
  if (dim % spec.block_dims != 0) {
    throw Error("quantize: dim (" + std::to_string(dim) + ") must be divisible by block_dims (" +
                std::to_string(spec.block_dims) + ")");
  }
}

std::size_t num_blocks(std::size_t dim, std::size_t block_dims) {
  validate_block_dims(block_dims);
  if (dim % block_dims != 0) {
    throw Error("num_blocks: dim must be divisible by block_dims");
  }
  return dim / block_dims;
}

std::size_t l0_bits_per_vector(std::size_t dim, std::size_t bits, std::size_t block_dims) {
  validate_bits_per_dim(bits);
  validate_block_dims(block_dims);
  if (dim % block_dims != 0) {
    throw Error("l0_bits_per_vector: dim must be divisible by block_dims");
  }
  return (dim / block_dims) * bits;
}

std::size_t l0_words_per_vector(std::size_t dim, std::size_t bits, std::size_t block_dims) {
  const std::size_t total_bits = l0_bits_per_vector(dim, bits, block_dims);
  return (total_bits + 63) / 64;
}

LloydMaxCodebook::LloydMaxCodebook(std::size_t dim, QuantSpec spec)
    : LloydMaxCodebook(dim, spec.bits, spec.block_dims) {}

LloydMaxCodebook::LloydMaxCodebook(std::size_t dim, std::size_t bits, std::size_t block_dims)
    : dim_(dim), bits_(bits), block_dims_(block_dims) {
  validate_quant_spec(dim_, QuantSpec{block_dims_, bits_});
  num_centroids_ = std::size_t{1} << bits_;

  if (block_dims_ == 1) {
    auto [boundaries, centroids] = lloyd_max_beta(bits_, dim_);
    boundaries_ = std::move(boundaries);
    centroids_ = std::move(centroids);
  } else {
    centroids_ = lloyd_max_beta_block(bits_, dim_, block_dims_);
    boundaries_.clear();
  }
}

std::span<const float> LloydMaxCodebook::centroid(std::uint32_t index) const {
  if (index >= num_centroids_) {
    throw Error("LloydMaxCodebook::centroid: index out of range");
  }
  return std::span<const float>(centroids_.data() + static_cast<std::size_t>(index) * block_dims_,
                                block_dims_);
}

float LloydMaxCodebook::centroid_at(std::uint32_t index) const {
  return centroid(index)[0];
}

std::uint32_t LloydMaxCodebook::encode(float x) const {
  if (block_dims_ != 1) {
    throw Error("LloydMaxCodebook::encode(float): requires block_dims == 1");
  }
  if (num_centroids_ == 0) {
    throw Error("LloydMaxCodebook::encode on empty codebook");
  }
  std::uint32_t idx = 0;
  while (idx < boundaries_.size() && x > boundaries_[idx]) {
    ++idx;
  }
  return idx;
}

std::uint32_t LloydMaxCodebook::encode(std::span<const float> block) const {
  if (block.size() != block_dims_) {
    throw Error("LloydMaxCodebook::encode(block): size must equal block_dims");
  }
  if (num_centroids_ == 0) {
    throw Error("LloydMaxCodebook::encode on empty codebook");
  }
  if (block_dims_ == 1) {
    return encode(block[0]);
  }
  if (bits_ == 1 && num_centroids_ == 2) {
    // Half-plane: nearer of two centroids.
    const float d0 = squared_dist(block, centroid(0));
    const float d1 = squared_dist(block, centroid(1));
    return d1 < d0 ? 1u : 0u;
  }
  return nearest_centroid(block, centroids_, num_centroids_, block_dims_);
}

std::uint32_t unpack_code(std::span<const std::uint64_t> words, std::size_t block_index,
                          std::size_t bits) {
  validate_bits_per_dim(bits);
  const std::size_t bit_pos = block_index * bits;
  const std::size_t word_i = bit_pos / 64;
  const std::size_t bit_i = bit_pos % 64;
  const std::uint64_t mask = (bits == 64) ? ~0ull : ((std::uint64_t{1} << bits) - 1ull);
  if (bit_i + bits <= 64) {
    if (word_i >= words.size()) {
      throw Error("unpack_code: word index out of range");
    }
    return static_cast<std::uint32_t>((words[word_i] >> bit_i) & mask);
  }
  if (word_i + 1 >= words.size()) {
    throw Error("unpack_code: straddling word index out of range");
  }
  const std::size_t lo_bits = 64 - bit_i;
  const std::uint64_t lo = words[word_i] >> bit_i;
  const std::uint64_t hi = words[word_i + 1] & ((std::uint64_t{1} << (bits - lo_bits)) - 1ull);
  return static_cast<std::uint32_t>((lo | (hi << lo_bits)) & mask);
}

std::size_t quantize_blocks_to_nbit_into(std::span<const float> vector,
                                         const LloydMaxCodebook& codebook,
                                         std::span<std::uint64_t> out) {
  const std::size_t dim = vector.size();
  const std::size_t bits = codebook.bits();
  const std::size_t block_dims = codebook.block_dims();
  if (dim != codebook.dim()) {
    throw Error("quantize_blocks_to_nbit_into: vector dim must match codebook dim");
  }
  const std::size_t need_words = l0_words_per_vector(dim, bits, block_dims);
  if (out.size() < need_words) {
    throw Error("quantize_blocks_to_nbit_into: output too small");
  }
  std::fill(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(need_words), 0);

  const std::uint64_t mask = (bits == 64) ? ~0ull : ((std::uint64_t{1} << bits) - 1ull);
  const std::size_t m = dim / block_dims;
  for (std::size_t b = 0; b < m; ++b) {
    const auto block = vector.subspan(b * block_dims, block_dims);
    const std::uint64_t code = static_cast<std::uint64_t>(codebook.encode(block)) & mask;
    pack_code(out, b * bits, bits, code);
  }
  return m * bits;
}

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_blocks_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook) {
  const std::size_t num_words =
      l0_words_per_vector(vector.size(), codebook.bits(), codebook.block_dims());
  std::vector<std::uint64_t> words(num_words, 0);
  const std::size_t num_bits = quantize_blocks_to_nbit_into(vector, codebook, words);
  return {std::move(words), num_bits};
}

std::size_t quantize_1dim_to_nbit_into(std::span<const float> vector, const LloydMaxCodebook& codebook,
                                       std::span<std::uint64_t> out) {
  return quantize_blocks_to_nbit_into(vector, codebook, out);
}

std::pair<std::vector<std::uint64_t>, std::size_t> quantize_1dim_to_nbit(
    std::span<const float> vector, const LloydMaxCodebook& codebook) {
  return quantize_blocks_to_nbit(vector, codebook);
}

float ip_scale_alpha(std::span<const float> rotated_unit, std::span<const std::uint64_t> codes,
                     const LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.dim();
  const std::size_t bits = codebook.bits();
  const std::size_t block_dims = codebook.block_dims();
  if (rotated_unit.size() != dim) {
    throw Error("ip_scale_alpha: rotated dim mismatch");
  }
  if (codes.size() < l0_words_per_vector(dim, bits, block_dims)) {
    throw Error("ip_scale_alpha: codes too small");
  }
  double ip = 0.0;
  const std::size_t m = dim / block_dims;
  for (std::size_t b = 0; b < m; ++b) {
    const std::uint32_t code = unpack_code(codes, b, bits);
    const auto c = codebook.centroid(code);
    for (std::size_t j = 0; j < block_dims; ++j) {
      ip += static_cast<double>(rotated_unit[b * block_dims + j]) * static_cast<double>(c[j]);
    }
  }
  if (std::abs(ip) < 1e-12) {
    return 0.0f;
  }
  // Unit vectors: ||v|| = 1 after L2 normalize → α = 1 / <u, x_hat>.
  return static_cast<float>(1.0 / ip);
}

}  // namespace vectorcache::quantize
