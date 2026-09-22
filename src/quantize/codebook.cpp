#include "vectorcache/quantize/codebook.hpp"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>

namespace vectorcache {
namespace {

// Regularized incomplete beta Ix(a,a) via continued fraction (symmetric case).
double log_gamma_stirling(double z) {
  // Lanczos approximation (g=7).
  static constexpr double c[9] = {0.99999999999980993, 676.5203681218851,   -1259.1392167224028,
                                  771.32342877765313,  -176.61502916214059, 12.507343278686905,
                                  -0.13857109526572012, 9.984369654078861e-6, 1.5056327351493116e-7};
  if (z < 0.5) {
    return std::log(3.14159265358979323846) - std::log(std::sin(3.14159265358979323846 * z)) -
           log_gamma_stirling(1.0 - z);
  }
  z -= 1.0;
  double x = c[0];
  for (int i = 1; i < 9; ++i) {
    x += c[i] / (z + static_cast<double>(i));
  }
  const double t = z + 7.5;
  return 0.5 * std::log(2.0 * 3.14159265358979323846) + (z + 0.5) * std::log(t) - t + std::log(x);
}

double betacf(double a, double b, double x) {
  constexpr int kMaxIt = 200;
  constexpr double kEps = 3e-14;
  constexpr double kFpmin = 1e-30;
  double qab = a + b;
  double qap = a + 1.0;
  double qam = a - 1.0;
  double c = 1.0;
  double d = 1.0 - qab * x / qap;
  if (std::fabs(d) < kFpmin) {
    d = kFpmin;
  }
  d = 1.0 / d;
  double h = d;
  for (int m = 1; m <= kMaxIt; ++m) {
    const int m2 = 2 * m;
    double aa = static_cast<double>(m) * (b - static_cast<double>(m)) * x /
                ((qam + m2) * (a + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    h *= d * c;
    aa = -(a + static_cast<double>(m)) * (qab + static_cast<double>(m)) * x /
         ((a + m2) * (qap + m2));
    d = 1.0 + aa * d;
    if (std::fabs(d) < kFpmin) {
      d = kFpmin;
    }
    c = 1.0 + aa / c;
    if (std::fabs(c) < kFpmin) {
      c = kFpmin;
    }
    d = 1.0 / d;
    const double del = d * c;
    h *= del;
    if (std::fabs(del - 1.0) < kEps) {
      break;
    }
  }
  return h;
}

double incomplete_beta(double a, double b, double x) {
  if (x <= 0.0) {
    return 0.0;
  }
  if (x >= 1.0) {
    return 1.0;
  }
  const double lbeta =
      log_gamma_stirling(a) + log_gamma_stirling(b) - log_gamma_stirling(a + b);
  const double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / a;
  if (x < (a + 1.0) / (a + b + 2.0)) {
    return front * betacf(a, b, x);
  }
  return 1.0 - (std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / b) * betacf(b, a, 1.0 - x);
}

struct BetaAA {
  double a;
  double pdf(double t) const {
    // Beta(a,a) pdf on [0,1]
    if (t <= 0.0 || t >= 1.0) {
      return 0.0;
    }
    const double lbeta = 2.0 * log_gamma_stirling(a) - log_gamma_stirling(2.0 * a);
    return std::exp((a - 1.0) * std::log(t) + (a - 1.0) * std::log(1.0 - t) - lbeta);
  }
  double cdf(double t) const { return incomplete_beta(a, a, t); }
};

template <typename F>
double adaptive_simpson_rec(F& f, double a, double b, double fa, double fb, double fm, double whole,
                            double tol, std::size_t depth) {
  const double mid = (a + b) / 2.0;
  const double m1 = (a + mid) / 2.0;
  const double m2 = (mid + b) / 2.0;
  const double fm1 = f(m1);
  const double fm2 = f(m2);
  const double left = (mid - a) / 6.0 * (fa + 4.0 * fm1 + fm);
  const double right = (b - mid) / 6.0 * (fm + 4.0 * fm2 + fb);
  const double refined = left + right;
  if (depth == 0 || std::fabs(refined - whole) < 15.0 * tol) {
    return refined + (refined - whole) / 15.0;
  }
  return adaptive_simpson_rec(f, a, mid, fa, fm, fm1, left, tol / 2.0, depth - 1) +
         adaptive_simpson_rec(f, mid, b, fm, fb, fm2, right, tol / 2.0, depth - 1);
}

template <typename F>
double adaptive_simpson(F f, double a, double b, double tol, std::size_t max_depth) {
  const double mid = (a + b) / 2.0;
  const double fa = f(a);
  const double fb = f(b);
  const double fm = f(mid);
  const double whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
  return adaptive_simpson_rec(f, a, b, fa, fb, fm, whole, tol, max_depth);
}

std::pair<std::vector<float>, std::vector<float>> lloyd_max(std::size_t bits, std::size_t dim,
                                                            std::size_t max_iter, double tol) {
  const double a = (static_cast<double>(dim) - 1.0) / 2.0;
  BetaAA beta{a};
  const std::size_t n_levels = 1u << bits;
  const double std_dev = std::sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a));
  const double spread = 3.0 * std_dev;
  std::vector<double> centroids(n_levels);
  for (std::size_t i = 0; i < n_levels; ++i) {
    centroids[i] = -spread + 2.0 * spread * static_cast<double>(i) / static_cast<double>(n_levels - 1);
  }

  for (std::size_t it = 0; it < max_iter; ++it) {
    std::vector<double> boundaries(n_levels - 1);
    for (std::size_t i = 0; i + 1 < n_levels; ++i) {
      boundaries[i] = (centroids[i] + centroids[i + 1]) / 2.0;
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
      const double cdf_lo = beta.cdf((lo + 1.0) / 2.0);
      const double cdf_hi = beta.cdf((hi + 1.0) / 2.0);
      const double prob = cdf_hi - cdf_lo;
      if (prob < 1e-15) {
        new_centroids[i] = centroids[i];
      } else {
        auto integrand = [&](double x) {
          const double t = (x + 1.0) / 2.0;
          return x * beta.pdf(t) / 2.0;
        };
        const double mean = adaptive_simpson(integrand, lo, hi, 1e-14, 50);
        new_centroids[i] = mean / prob;
      }
    }
    double max_change = 0.0;
    for (std::size_t i = 0; i < n_levels; ++i) {
      max_change = std::max(max_change, std::fabs(centroids[i] - new_centroids[i]));
    }
    centroids = std::move(new_centroids);
    if (max_change < tol) {
      break;
    }
  }

  std::vector<float> centroids_f32(n_levels);
  for (std::size_t i = 0; i < n_levels; ++i) {
    centroids_f32[i] = static_cast<float>(centroids[i]);
  }
  std::vector<float> boundaries_f32(n_levels - 1);
  for (std::size_t i = 0; i + 1 < n_levels; ++i) {
    boundaries_f32[i] = (centroids_f32[i] + centroids_f32[i + 1]) * 0.5f;
  }
  return {std::move(boundaries_f32), std::move(centroids_f32)};
}

std::mutex g_memo_mu;
std::map<std::pair<std::size_t, std::size_t>, std::pair<std::vector<float>, std::vector<float>>>
    g_memo;

}  // namespace

std::pair<std::vector<float>, std::vector<float>> codebook(std::size_t bits, std::size_t dim) {
  if (bits < 2 || bits > 4) {
    throw std::invalid_argument("bits must be 2, 3, or 4");
  }
  if (dim < 2) {
    throw std::invalid_argument("dim must be >= 2");
  }
  {
    std::unique_lock lock(g_memo_mu, std::try_to_lock);
    if (lock.owns_lock()) {
      auto it = g_memo.find({bits, dim});
      if (it != g_memo.end()) {
        return it->second;
      }
    }
  }
  auto computed = lloyd_max(bits, dim, 200, 1e-12);
  {
    std::unique_lock lock(g_memo_mu, std::try_to_lock);
    if (lock.owns_lock()) {
      g_memo.emplace(std::make_pair(bits, dim), computed);
    }
  }
  return computed;
}

}  // namespace vectorcache
