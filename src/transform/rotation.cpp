#include "vectorcache/transform/rotation.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <stdexcept>

#include "vectorcache/chacha8.hpp"

namespace vectorcache {
namespace {

constexpr std::array<std::uint8_t, 32> kRotationSeed = {
    164, 143, 161, 123, 88,  50,  61,  10,  234, 184, 161, 204, 105, 1,   20,  184,
    43,  140, 200, 117, 24,  180, 247, 84,  141, 68,  110, 161, 228, 223, 32,  242};

std::size_t block_size(std::size_t dim) { return dim & (~dim + 1); }

std::vector<std::uint32_t> fisher_yates(std::size_t dim, ChaCha8Rng& rng) {
  std::vector<std::uint32_t> perm(dim);
  for (std::size_t i = 0; i < dim; ++i) {
    perm[i] = static_cast<std::uint32_t>(i);
  }
  for (std::size_t i = dim; i-- > 1;) {
    const std::size_t j = static_cast<std::size_t>(rng.next_u64() % (static_cast<std::uint64_t>(i) + 1));
    std::swap(perm[i], perm[j]);
  }
  return perm;
}

void wht_block_scalar(float* blk, std::size_t block, float inv_sqrt_block) {
  for (std::size_t len = 1; len < block; len <<= 1) {
    for (std::size_t i = 0; i < block; i += 2 * len) {
      for (std::size_t j = i; j < i + len; ++j) {
        const float a = blk[j];
        const float b = blk[j + len];
        blk[j] = a + b;
        blk[j + len] = a - b;
      }
    }
  }
  for (std::size_t i = 0; i < block; ++i) {
    blk[i] *= inv_sqrt_block;
  }
}

void wht_all(float* buf, std::size_t dim, std::size_t block, float inv_sqrt_block) {
  for (std::size_t offset = 0; offset < dim; offset += block) {
    wht_block_scalar(buf + offset, block, inv_sqrt_block);
  }
}

template <int Mode>
void permute_gather(const float* src, const std::uint32_t* perm, const float* signs, float inv,
                    float* dst, std::size_t dim) {
  for (std::size_t i = 0; i < dim; ++i) {
    const float v = src[perm[i]];
    if constexpr (Mode == 0) {
      dst[i] = v;
    } else if constexpr (Mode == 1) {
      dst[i] = v * signs[i];
    } else {
      dst[i] = (v * inv) * signs[i];
    }
  }
}

}  // namespace

Rotation::Rotation(std::size_t dim) : dim_(dim) {
  if (dim == 0 || dim % 8 != 0) {
    throw std::invalid_argument("rotation dim must be a positive multiple of 8");
  }
  if (dim > kMaxDim) {
    throw std::invalid_argument("rotation dim exceeds MAX_DIM");
  }
  block_ = block_size(dim);
  inv_sqrt_block_ = 1.f / std::sqrt(static_cast<float>(block_));

  ChaCha8Rng rng(kRotationSeed);
  signs_.resize(kRotationRounds);
  perms_.resize(kRotationRounds);
  for (std::size_t r = 0; r < kRotationRounds; ++r) {
    signs_[r].resize(dim);
    for (std::size_t i = 0; i < dim; ++i) {
      signs_[r][i] = (rng.next_u32() & 1u) ? -1.f : 1.f;
    }
    perms_[r] = fisher_yates(dim, rng);
  }
  signs1_pre_.assign(dim, 1.f);
  for (std::size_t i = 0; i < dim; ++i) {
    signs1_pre_[perms_[1][i]] = signs_[1][i];
  }
}

void Rotation::apply(std::span<float> row) const {
  std::vector<float> scratch(dim_);
  apply_with_scratch(row, scratch);
}

void Rotation::apply_with_scratch(std::span<float> row, std::span<float> scratch) const {
  assert(row.size() == dim_ && scratch.size() == dim_);
  float* a = row.data();
  float* b = scratch.data();
  for (std::size_t round = 0; round < kRotationRounds; ++round) {
    permute_gather<1>(a, perms_[round].data(), signs_[round].data(), 1.f, b, dim_);
    wht_all(b, dim_, block_, inv_sqrt_block_);
    std::swap(a, b);
  }
  // Even K: result is back in `row`. If a swapped away from row, copy.
  if (a != row.data()) {
    std::copy(a, a + static_cast<std::ptrdiff_t>(dim_), row.begin());
  }
}

void Rotation::apply_scaled_into(std::span<const float> src, float inv, std::span<float> dst,
                                 std::span<float> scratch) const {
  assert(src.size() == dim_ && dst.size() == dim_ && scratch.size() == dim_);
  permute_gather<2>(src.data(), perms_[0].data(), signs_[0].data(), inv, scratch.data(), dim_);
  wht_all(scratch.data(), dim_, block_, inv_sqrt_block_);
  for (std::size_t i = 0; i < dim_; ++i) {
    scratch[i] *= signs1_pre_[i];
  }
  permute_gather<0>(scratch.data(), perms_[1].data(), signs1_pre_.data(), 1.f, dst.data(), dim_);
  wht_all(dst.data(), dim_, block_, inv_sqrt_block_);
}

}  // namespace vectorcache
