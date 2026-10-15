#include "vectorcache/transform/landmarks.hpp"

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace vectorcache {

std::vector<float> make_ortho_landmarks(std::size_t dim, std::size_t m, std::uint64_t seed) {
  if (dim == 0 || m == 0 || m > dim) {
    throw std::invalid_argument("landmarks require 0 < m <= dim");
  }
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> A(static_cast<std::size_t>(dim) * m);
  for (double& v : A) {
    v = nd(rng);
  }

  // Modified Gram-Schmidt on columns (column-major).
  for (std::size_t i = 0; i < m; ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      double dot = 0.0;
      for (std::size_t r = 0; r < dim; ++r) {
        dot += A[r + i * dim] * A[r + j * dim];
      }
      for (std::size_t r = 0; r < dim; ++r) {
        A[r + i * dim] -= dot * A[r + j * dim];
      }
    }
    double nrm = 0.0;
    for (std::size_t r = 0; r < dim; ++r) {
      const double v = A[r + i * dim];
      nrm += v * v;
    }
    nrm = std::sqrt(nrm);
    if (nrm < 1e-15) {
      throw std::runtime_error("degenerate landmark column");
    }
    const double inv = 1.0 / nrm;
    for (std::size_t r = 0; r < dim; ++r) {
      A[r + i * dim] *= inv;
    }
  }

  std::vector<float> L(static_cast<std::size_t>(dim) * m);
  for (std::size_t i = 0; i < A.size(); ++i) {
    L[i] = static_cast<float>(A[i]);
  }
  return L;
}

}  // namespace vectorcache
