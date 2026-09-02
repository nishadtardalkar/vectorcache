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

bool use_stockham(std::size_t n) { return n == 256 || n == 2048 || n == 4096; }

}  // namespace

SrhtRotation::SrhtRotation(std::size_t original_dim, std::uint64_t seed)
    : original_dim_(original_dim),
      padded_dim_(vectorcache::transform::padded_dim(original_dim)),
      inv_sqrt_n_(1.0f / std::sqrt(static_cast<float>(padded_dim_))) {
  if (original_dim == 0) {
    throw Error("original_dim must be > 0");
  }
  std::mt19937_64 rng(seed);
#if VECTORCACHE_SRHT_ROUNDS >= 1
  signs1_f_ = rademacher_f(rng, padded_dim_);
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 2
  signs2_f_ = rademacher_f(rng, padded_dim_);
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 3
  signs3_f_ = rademacher_f(rng, padded_dim_);
#endif
  if (use_stockham(padded_dim_)) {
    fwht_scratch_.assign(padded_dim_, 0.0f);
  }
}

void SrhtRotation::apply_rounds(std::span<float> buf) const {
  const auto fwht = [this](std::span<float> span) {
    if (use_stockham(padded_dim_)) {
      fwht_stockham_orthonormal_in_place(span, fwht_scratch_, inv_sqrt_n_);
    } else {
      fwht_orthonormal_in_place(span, inv_sqrt_n_);
    }
  };

#if VECTORCACHE_SRHT_ROUNDS >= 1
  apply_signs_f(buf, signs1_f_);
  fwht(buf);
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 2
  apply_signs_f(buf, signs2_f_);
  fwht(buf);
#endif
#if VECTORCACHE_SRHT_ROUNDS >= 3
  apply_signs_f(buf, signs3_f_);
  fwht(buf);
#endif
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
