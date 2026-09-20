#include "vectorcache/transform/srht.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <utility>

#include "vectorcache/error.hpp"
#include "vectorcache/simd.hpp"
#include "vectorcache/transform/fwht.hpp"

namespace vectorcache::transform {

namespace {

/// Largest power-of-two divisor of dim (TurboVec block_size).
std::size_t hadamard_block_size(std::size_t dim) {
  if (dim == 0) {
    return 0;
  }
  return dim & (0u - dim);  // lowest set bit
}

std::vector<std::uint32_t> fisher_yates(std::size_t n, std::mt19937_64& rng) {
  std::vector<std::uint32_t> perm(n);
  for (std::size_t i = 0; i < n; ++i) {
    perm[i] = static_cast<std::uint32_t>(i);
  }
  for (std::size_t i = n; i > 1; --i) {
    std::uniform_int_distribution<std::size_t> dist(0, i - 1);
    const std::size_t j = dist(rng);
    std::swap(perm[i - 1], perm[j]);
  }
  return perm;
}

AlignedVector<float> rademacher_f(std::mt19937_64& rng, std::size_t n) {
  AlignedVector<float> signs(n);
  std::uniform_int_distribution<int> dist(0, 1);
  for (std::size_t i = 0; i < n; ++i) {
    signs[i] = dist(rng) == 0 ? -1.0f : 1.0f;
  }
  return signs;
}

void permute_gather_signed(std::span<const float> src, std::span<const std::uint32_t> perm,
                           std::span<const float> signs, std::span<float> dst) {
  const std::size_t n = dst.size();
  std::size_t i = 0;
  for (; i + simd::kWidth <= n; i += simd::kWidth) {
    const __m512i idx = _mm512_loadu_si512(perm.data() + i);
    const __m512 vals = _mm512_i32gather_ps(idx, src.data(), 4);
    const __m512 sgn = _mm512_load_ps(signs.data() + i);
    _mm512_storeu_ps(dst.data() + i, _mm512_mul_ps(vals, sgn));
  }
  for (; i < n; ++i) {
    dst[i] = src[perm[i]] * signs[i];
  }
}

void wht_blocks(std::span<float> buf, std::size_t block, float inv_sqrt_block) {
  if (block <= 1) {
    return;
  }
  for (std::size_t offset = 0; offset < buf.size(); offset += block) {
    fwht_orthonormal_in_place(buf.subspan(offset, block), inv_sqrt_block);
  }
}

}  // namespace

SrhtRotation::SrhtRotation(std::size_t original_dim, std::uint64_t seed)
    : dim_(original_dim),
      block_(hadamard_block_size(original_dim)),
      inv_sqrt_block_(block_ > 0 ? 1.0f / std::sqrt(static_cast<float>(block_)) : 1.0f) {
  if (original_dim == 0) {
    throw Error("original_dim must be > 0");
  }
  if (block_ == 0 || (original_dim % block_) != 0) {
    throw Error("SrhtRotation: invalid block size for dim");
  }

  constexpr std::size_t kRounds = VECTORCACHE_SRHT_ROUNDS;
  signs_.resize(kRounds);
  perms_.resize(kRounds);
  std::mt19937_64 rng(seed);
  for (std::size_t r = 0; r < kRounds; ++r) {
    signs_[r] = rademacher_f(rng, dim_);
    perms_[r] = fisher_yates(dim_, rng);
  }
}

void SrhtRotation::apply_rounds(std::span<float> buf) const {
  constexpr std::size_t kRounds = VECTORCACHE_SRHT_ROUNDS;
  thread_local AlignedVector<float> scratch;
  if (scratch.size() != dim_) {
    scratch.assign(dim_, 0.0f);
  }

  // Ping-pong: round r gathers into the other buffer, then block-WHT in place.
  float* a = buf.data();
  float* b = scratch.data();
  for (std::size_t r = 0; r < kRounds; ++r) {
    permute_gather_signed(std::span<const float>(a, dim_), perms_[r], signs_[r],
                          std::span<float>(b, dim_));
    wht_blocks(std::span<float>(b, dim_), block_, inv_sqrt_block_);
    std::swap(a, b);
  }
  // Even rounds: result in buf (a == buf). Odd rounds: result in scratch → copy back.
  if (kRounds % 2 == 1) {
    std::memcpy(buf.data(), scratch.data(), dim_ * sizeof(float));
  }
}

void SrhtRotation::apply(std::span<const float> vector, std::span<float> out) const {
  if (vector.size() != dim_) {
    throw Error("vector dimension mismatch: expected " + std::to_string(dim_) + ", got " +
                std::to_string(vector.size()));
  }
  if (out.size() != dim_) {
    throw Error("output buffer dimension mismatch: expected " + std::to_string(dim_) + ", got " +
                std::to_string(out.size()));
  }
  std::memcpy(out.data(), vector.data(), dim_ * sizeof(float));
  apply_rounds(out);
}

void SrhtRotation::apply_in_place(std::span<float> buf) const {
  if (buf.size() != dim_) {
    throw Error("buffer dimension mismatch: expected " + std::to_string(dim_) + ", got " +
                std::to_string(buf.size()));
  }
  apply_rounds(buf);
}

}  // namespace vectorcache::transform
