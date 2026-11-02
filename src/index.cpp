#include "vectorcache/index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "vectorcache/encode/encode.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/search/float_rescore.hpp"
#include "vectorcache/transform/landmarks.hpp"

namespace vectorcache {
namespace {

float l2_normalize_into(const float* src, float* dst, std::size_t dim) {
  double s = 0.0;
#if defined(__x86_64__) || defined(_M_X64)
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 v = _mm256_loadu_ps(src + i);
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  s = static_cast<double>(tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]);
  for (; i < dim; ++i) {
    s += static_cast<double>(src[i]) * src[i];
  }
#else
  for (std::size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(src[i]) * src[i];
  }
#endif
  const float nrm = static_cast<float>(std::sqrt(s));
  if (nrm < kMinInputNorm) {
    std::fill(dst, dst + static_cast<std::ptrdiff_t>(dim), 0.f);
    return 0.f;
  }
  const float inv = 1.f / nrm;
  for (std::size_t d = 0; d < dim; ++d) {
    dst[d] = src[d] * inv;
  }
  return nrm;
}

/// Branchless level count — vectorizes cleanly for bits ∈ {2,3,4}.
inline std::size_t quantize_coord(float x, std::span<const float> boundaries) {
  std::size_t v = 0;
  for (float b : boundaries) {
    v += static_cast<std::size_t>(x >= b);
  }
  return v;
}

inline float norm_sq_f32(const float* x, std::size_t dim) {
#if defined(__x86_64__) || defined(_M_X64)
  __m256 acc = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 <= dim; i += 8) {
    const __m256 v = _mm256_loadu_ps(x + i);
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  alignas(32) float tmp[8];
  _mm256_store_ps(tmp, acc);
  float s = tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7];
  for (; i < dim; ++i) {
    s += x[i] * x[i];
  }
  return s;
#else
  float s = 0.f;
  for (std::size_t i = 0; i < dim; ++i) {
    s += x[i] * x[i];
  }
  return s;
#endif
}

std::size_t pad_scan_dim(std::size_t m) { return (m + 7u) & ~std::size_t{7u}; }

/// Encode one row into α·ŷ (full-dim) and return ‖α·ŷ‖². Uses caller scratch.
float encode_scaled_yhat(const float* src, float* unit_scratch, float* rot_scratch,
                         float* yhat_out, std::size_t dim, const Rotation& rotation,
                         std::span<const float> boundaries, std::span<const float> centroids) {
  l2_normalize_into(src, unit_scratch, dim);
  std::copy(unit_scratch, unit_scratch + static_cast<std::ptrdiff_t>(dim), rot_scratch);
  rotation.apply_with_scratch(std::span<float>(rot_scratch, dim),
                              std::span<float>(unit_scratch, dim));

  double dot = 0.0;
  double yhat_norm2 = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    const std::size_t code = quantize_coord(rot_scratch[i], boundaries);
    const float c = centroids[code];
    yhat_out[i] = c;
    dot += static_cast<double>(rot_scratch[i]) * c;
    yhat_norm2 += static_cast<double>(c) * c;
  }
  const float alpha =
      (yhat_norm2 > kDegenerateInnerEps) ? static_cast<float>(dot / yhat_norm2) : 0.f;

  float v_norm2 = 0.f;
  for (std::size_t i = 0; i < dim; ++i) {
    const float vi = alpha * yhat_out[i];
    yhat_out[i] = vi;
    v_norm2 += vi * vi;
  }
  return v_norm2;
}

}  // namespace

LandmarkIndex::LandmarkIndex(std::size_t dim, std::size_t scan_bits, std::size_t n_landmarks,
                             std::uint64_t landmark_seed, std::size_t rerank_bits)
    : dim_(dim),
      scan_bits_(scan_bits),
      rerank_bits_(rerank_bits == 0 ? scan_bits : rerank_bits),
      n_landmarks_(n_landmarks),
      scan_dim_(pad_scan_dim(n_landmarks)),
      landmark_seed_(landmark_seed),
      rotation_(dim) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("dim must be a multiple of 8 in [8, MAX_DIM]");
  }
  if (scan_bits_ < 2 || scan_bits_ > 4) {
    throw std::invalid_argument("scan_bits must be 2, 3, or 4");
  }
  if (rerank_bits_ < 2 || rerank_bits_ > 4) {
    throw std::invalid_argument("rerank_bits must be 2, 3, or 4");
  }
  if (n_landmarks == 0 || n_landmarks > dim) {
    throw std::invalid_argument("n_landmarks must be in 1..dim");
  }
  auto cb = codebook(rerank_bits_, dim_);
  boundaries_ = std::move(cb.first);
  centroids_ = std::move(cb.second);
  auto lm_cb = codebook(scan_bits_, scan_dim_);
  lm_boundaries_ = std::move(lm_cb.first);
  lm_centroids_ = std::move(lm_cb.second);
  landmark_basis_ = make_ortho_landmarks(dim_, n_landmarks_, landmark_seed_);
  landmark_basis_q4_ = quantize_landmarks(landmark_basis_, dim_, n_landmarks_);
}

void LandmarkIndex::add(std::span<const float> vectors) {
  if (vectors.size() % dim_ != 0) {
    throw std::invalid_argument("vectors length must be multiple of dim");
  }
  const std::size_t n = vectors.size() / dim_;
  if (n == 0) {
    return;
  }
  const std::size_t base = n_vectors_;
  n_vectors_ += n;
  landmark_rows_.resize(n_vectors_ * scan_dim_);
  residual_norms_.resize(n_vectors_);
  prepared_ = false;

  // Process in row tiles so α·ŷ fits in cache while L is reused across the tile.
  constexpr std::size_t kTile = 64;
  const float* L = landmark_basis_.data();
  const std::size_t dim = dim_;
  const std::size_t m = n_landmarks_;
  const std::size_t scan_dim = scan_dim_;
  const auto boundaries = std::span<const float>(boundaries_);
  const auto centroids = std::span<const float>(centroids_);
  const Rotation& rotation = rotation_;

  for (std::size_t tile = 0; tile < n; tile += kTile) {
    const std::size_t tile_n = std::min(kTile, n - tile);
    std::vector<float> Y(tile_n * dim);
    std::vector<float> v_norm2(tile_n);

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      std::vector<float> unit_scratch(dim);
      std::vector<float> rot_scratch(dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int ii = 0; ii < static_cast<int>(tile_n); ++ii) {
        const std::size_t i = static_cast<std::size_t>(ii);
        const float* src = vectors.data() + (tile + i) * dim;
        float* yhat = Y.data() + i * dim;
        v_norm2[i] =
            encode_scaled_yhat(src, unit_scratch.data(), rot_scratch.data(), yhat, dim, rotation,
                               boundaries, centroids);
      }
    }

    float* lm_out = landmark_rows_.data() + (base + tile) * scan_dim;
    project_rows_onto_landmarks(Y.data(), tile_n, dim, L, m, lm_out, scan_dim);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int ii = 0; ii < static_cast<int>(tile_n); ++ii) {
      const std::size_t i = static_cast<std::size_t>(ii);
      const float* row = lm_out + i * scan_dim;
      const float land_norm2 = norm_sq_f32(row, m);
      residual_norms_[base + tile + i] =
          std::sqrt(std::max(0.f, v_norm2[i] - land_norm2));
    }
  }
}

void LandmarkIndex::prepare() {
  // L2-normalize landmark rows for TurboQuant-style α, then FastScan-pack.
  std::vector<float> unit_lm(n_vectors_ * scan_dim_);
  std::vector<float> norms(n_vectors_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ii = 0; ii < static_cast<int>(n_vectors_); ++ii) {
    const std::size_t i = static_cast<std::size_t>(ii);
    const float* src = landmark_rows_.data() + i * scan_dim_;
    float* dst = unit_lm.data() + i * scan_dim_;
    norms[i] = l2_normalize_into(src, dst, scan_dim_);
  }

  std::vector<std::uint8_t> packed;
  scales_.clear();
  encode_prerotated(unit_lm, norms, n_vectors_, scan_dim_, lm_boundaries_, lm_centroids_,
                    scan_bits_, nullptr, packed, scales_);

  // Single extract → pack; reuse flat codes for planes densify path.
  const std::size_t n_byte_groups = scan_dim_ / (8 / scan_bits_);
  codes_flat_ = codes_flat_from_packed(packed, n_vectors_, scan_bits_, scan_dim_);
  auto packed_search = pack_flat_for_search(codes_flat_, n_vectors_, scan_bits_, n_byte_groups);
  blocked_codes_ = std::move(packed_search.first);
  n_blocks_ = packed_search.second;

  sign_codes_.clear();
  sign_n_blocks_ = 0;
  if (planes_wanted(scan_bits_, n_byte_groups, n_vectors_)) {
    auto sign = planes_repack_sign(packed, n_vectors_, scan_bits_, scan_dim_);
    sign_codes_ = std::move(sign.first);
    sign_n_blocks_ = sign.second;
  } else {
    codes_flat_.clear();
    codes_flat_.shrink_to_fit();
  }

  // Pack α and ‖r‖ as AoS pairs so flush loads one stream.
  scale_resid_.resize(2 * n_vectors_);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int ii = 0; ii < static_cast<int>(n_vectors_); ++ii) {
    const std::size_t i = static_cast<std::size_t>(ii);
    scale_resid_[2 * i] = scales_[i];
    scale_resid_[2 * i + 1] = residual_norms_[i];
  }
  landmark_rows_.clear();
  landmark_rows_.shrink_to_fit();
  prepared_ = true;
}

SearchResults LandmarkIndex::search(std::span<const float> queries, std::size_t k,
                                    std::span<const float> float_corpus) const {
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("queries length must be multiple of dim");
  }
  if (!prepared_) {
    throw std::runtime_error("call prepare() before search");
  }
  const std::size_t nq = queries.size() / dim_;
  const std::size_t kk = std::min(k, n_vectors_);
  SearchResults out =
      search_landmark_rerank(queries, nq, dim_, kk, landmark_basis_q4_, scan_dim_, lm_centroids_,
                             scan_bits_, blocked_codes_, n_blocks_, scales_, residual_norms_,
                             rotation_, scale_resid_, sign_codes_, sign_n_blocks_, codes_flat_);
  if (!float_corpus.empty()) {
    if (float_corpus.size() < n_vectors_ * dim_) {
      throw std::invalid_argument("float_corpus must have at least size() * dim rows");
    }
    // Exact IP on the ANN shortlist; caller should include this in timed search.
    float_rescore_topk(out, queries, float_corpus.subspan(0, n_vectors_ * dim_), dim_);
  }
  return out;
}

}  // namespace vectorcache
