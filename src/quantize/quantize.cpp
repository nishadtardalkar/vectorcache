#include "vectorcache/quantize/quantize.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
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

LloydMaxCodebook::LloydMaxCodebook(std::size_t dim, std::size_t bits) : dim_(dim), bits_(bits) {
  if (dim_ < 2) {
    throw Error("LloydMaxCodebook: dim must be >= 2");
  }
  validate_bits_per_dim(bits_);

  auto [boundaries, centroids] = lloyd_max_beta(bits_, dim_);
  boundaries_ = std::move(boundaries);
  centroids_ = std::move(centroids);
}

std::uint32_t LloydMaxCodebook::encode(float x) const {
  const std::size_t k = centroids_.size();
  if (k == 0) {
    throw Error("LloydMaxCodebook::encode on empty codebook");
  }
  // Clamp to support region [-1, 1]; extremes map to edge codes.
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
  if (dim != codebook.dim()) {
    throw Error("quantize_1dim_to_nbit_into: vector dim must match codebook dim");
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

float ip_scale_alpha(std::span<const float> rotated_unit, std::span<const std::uint64_t> codes,
                     const LloydMaxCodebook& codebook) {
  const std::size_t dim = codebook.dim();
  const std::size_t bits = codebook.bits();
  if (rotated_unit.size() != dim) {
    throw Error("ip_scale_alpha: rotated dim mismatch");
  }
  if (codes.size() < l0_words_per_vector(dim, bits)) {
    throw Error("ip_scale_alpha: codes too small");
  }
  double ip = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    const std::uint32_t code = unpack_code(codes, d, bits);
    ip += static_cast<double>(rotated_unit[d]) * static_cast<double>(codebook.centroid_at(code));
  }
  if (std::abs(ip) < 1e-12) {
    return 0.0f;
  }
  // Unit vectors: ||v|| = 1 after L2 normalize → α = 1 / <u, x_hat>.
  return static_cast<float>(1.0 / ip);
}

}  // namespace vectorcache::quantize
