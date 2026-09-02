#include "vectorcache/transform/srht.hpp"

#include <cmath>
#include <cstring>
#include <random>

#include "vectorcache/aligned.hpp"
#include "vectorcache/error.hpp"
#include "vectorcache/transform/fwht.hpp"

namespace vectorcache::transform {

namespace {

AlignedVector<float> rademacher_f(std::mt19937_64& rng, std::size_t n) {
  AlignedVector<float> signs(n);
  std::uniform_int_distribution<int> dist(0, 1);
  for (std::size_t i = 0; i < n; ++i) {
    signs[i] = dist(rng) == 0 ? -1.0f : 1.0f;
  }
  return signs;
}

}  // namespace

SrhtRotation::SrhtRotation(std::size_t original_dim, std::uint64_t seed)
    : original_dim_(original_dim),
      padded_dim_(vectorcache::transform::padded_dim(original_dim)),
      inv_sqrt_n_(1.0f / std::sqrt(static_cast<float>(padded_dim_))) {
  if (original_dim == 0) {
    throw Error("original_dim must be > 0");
  }
  std::mt19937_64 rng(seed);
  signs1_f_ = rademacher_f(rng, padded_dim_);
  signs2_f_ = rademacher_f(rng, padded_dim_);
  signs3_f_ = rademacher_f(rng, padded_dim_);
}

void SrhtRotation::apply_rounds(std::span<float> buf) const {
  apply_signs_f(buf, signs1_f_);
  fwht_orthonormal_in_place(buf, inv_sqrt_n_);

  apply_signs_f(buf, signs2_f_);
  fwht_orthonormal_in_place(buf, inv_sqrt_n_);

  apply_signs_f(buf, signs3_f_);
  fwht_orthonormal_in_place(buf, inv_sqrt_n_);
}

void SrhtRotation::apply(std::span<const float> vector, std::span<float> out) const {
  if (vector.size() != original_dim_) {
    throw Error("vector dimension mismatch: expected " + std::to_string(original_dim_) +
                ", got " + std::to_string(vector.size()));
  }
  if (out.size() != padded_dim_) {
    throw Error("output buffer dimension mismatch: expected " + std::to_string(padded_dim_) +
                ", got " + std::to_string(out.size()));
  }

  std::memcpy(out.data(), vector.data(), original_dim_ * sizeof(float));
  if (padded_dim_ > original_dim_) {
    std::memset(out.data() + original_dim_, 0, (padded_dim_ - original_dim_) * sizeof(float));
  }

  apply_rounds(out);
}

void SrhtRotation::apply_in_place(std::span<float> buf) const {
  if (buf.size() != padded_dim_) {
    throw Error("buffer dimension mismatch: expected " + std::to_string(padded_dim_) + ", got " +
                std::to_string(buf.size()));
  }
  apply_rounds(buf);
}

}  // namespace vectorcache::transform
