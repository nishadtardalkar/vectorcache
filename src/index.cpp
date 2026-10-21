#include "vectorcache/index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vectorcache/encode/encode.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/transform/landmarks.hpp"

namespace vectorcache {
namespace {

float l2_normalize_into(const float* src, float* dst, std::size_t dim) {
  double s = 0.0;
  for (std::size_t i = 0; i < dim; ++i) {
    s += static_cast<double>(src[i]) * src[i];
  }
  const float nrm = static_cast<float>(std::sqrt(s));
  if (nrm < kMinInputNorm) {
    for (std::size_t i = 0; i < dim; ++i) {
      dst[i] = 0.f;
    }
    return 0.f;
  }
  const float inv = 1.f / nrm;
  for (std::size_t i = 0; i < dim; ++i) {
    dst[i] = src[i] * inv;
  }
  return nrm;
}

std::size_t quantize_coord(float x, std::span<const float> boundaries) {
  std::size_t lo = 0;
  std::size_t hi = boundaries.size();
  while (lo < hi) {
    const std::size_t mid = lo + (hi - lo) / 2;
    if (x < boundaries[mid]) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }
  return lo;
}

std::size_t pad_scan_dim(std::size_t m) { return (m + 7u) & ~std::size_t{7u}; }

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
  rotate_scratch_.resize(dim_);
}

void LandmarkIndex::encode_row(const float* src, float* unit_out, float* landmark_row_out,
                               float* residual_out) {
  l2_normalize_into(src, unit_out, dim_);
  std::copy(unit_out, unit_out + static_cast<std::ptrdiff_t>(dim_), rotate_scratch_.begin());
  rotation_.apply(rotate_scratch_);

  double dot = 0.0;
  double yhat_norm2 = 0.0;
  std::vector<float> yhat(dim_);
  for (std::size_t i = 0; i < dim_; ++i) {
    const std::size_t code = quantize_coord(rotate_scratch_[i], boundaries_);
    const float c = centroids_[code];
    yhat[i] = c;
    dot += static_cast<double>(rotate_scratch_[i]) * c;
    yhat_norm2 += static_cast<double>(c) * c;
  }
  const float alpha =
      (yhat_norm2 > kDegenerateInnerEps) ? static_cast<float>(dot / yhat_norm2) : 0.f;

  std::fill(landmark_row_out, landmark_row_out + scan_dim_, 0.f);
  double land_norm2 = 0.0;
  double v_norm2 = 0.0;
  for (std::size_t i = 0; i < dim_; ++i) {
    const double vi = static_cast<double>(alpha) * yhat[i];
    v_norm2 += vi * vi;
  }
  for (std::size_t j = 0; j < n_landmarks_; ++j) {
    double s = 0.0;
    const float* Lj = landmark_basis_.data() + j * dim_;
    for (std::size_t i = 0; i < dim_; ++i) {
      s += static_cast<double>(alpha * yhat[i]) * Lj[i];
    }
    landmark_row_out[j] = static_cast<float>(s);
    land_norm2 += s * s;
  }
  // ‖r‖ = ‖(α·ŷ) - P_L(α·ŷ)‖ with orthonormal L columns.
  *residual_out = static_cast<float>(std::sqrt(std::max(0.0, v_norm2 - land_norm2)));
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
  db_unit_.resize(n_vectors_ * dim_);
  landmark_rows_.resize(n_vectors_ * scan_dim_);
  residual_norms_.resize(n_vectors_);
  prepared_ = false;

  for (std::size_t i = 0; i < n; ++i) {
    encode_row(vectors.data() + i * dim_, db_unit_.data() + (base + i) * dim_,
               landmark_rows_.data() + (base + i) * scan_dim_, &residual_norms_[base + i]);
  }
}

void LandmarkIndex::prepare() {
  // Drop unit floats — single-scan path does not keep full-dim re-rank codes.
  db_unit_.clear();
  db_unit_.shrink_to_fit();

  // L2-normalize landmark rows for TurboQuant-style α, then FastScan-pack.
  std::vector<float> unit_lm(n_vectors_ * scan_dim_);
  std::vector<float> norms(n_vectors_);
  for (std::size_t i = 0; i < n_vectors_; ++i) {
    const float* src = landmark_rows_.data() + i * scan_dim_;
    float* dst = unit_lm.data() + i * scan_dim_;
    norms[i] = l2_normalize_into(src, dst, scan_dim_);
  }

  std::vector<std::uint8_t> packed;
  scales_.clear();
  encode_prerotated(unit_lm, norms, n_vectors_, scan_dim_, lm_boundaries_, lm_centroids_,
                    scan_bits_, nullptr, packed, scales_);
  auto repacked = repack(packed, n_vectors_, scan_bits_, scan_dim_);
  blocked_codes_ = std::move(repacked.first);
  n_blocks_ = repacked.second;

  // Optional planes layout: keep full blocked_codes_ for exact rescore; add sign region
  // for the staged shortlist (same bytes turbovec scans with the 2-bit-shaped VNNI path).
  sign_codes_.clear();
  sign_n_blocks_ = 0;
  const std::size_t n_byte_groups = scan_dim_ / (8 / scan_bits_);
  if (planes_wanted(scan_bits_, n_byte_groups, n_vectors_)) {
    auto sign = planes_repack_sign(packed, n_vectors_, scan_bits_, scan_dim_);
    sign_codes_ = std::move(sign.first);
    sign_n_blocks_ = sign.second;
  }

  // Pack α and ‖r‖ as AoS pairs so flush loads one stream.
  scale_resid_.resize(2 * n_vectors_);
  for (std::size_t i = 0; i < n_vectors_; ++i) {
    scale_resid_[2 * i] = scales_[i];
    scale_resid_[2 * i + 1] = residual_norms_[i];
  }
  landmark_rows_.clear();
  landmark_rows_.shrink_to_fit();
  prepared_ = true;
}

SearchResults LandmarkIndex::search(std::span<const float> queries, std::size_t k,
                                    SearchStageTimings* timings) const {
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("queries length must be multiple of dim");
  }
  if (!prepared_) {
    throw std::runtime_error("call prepare() before search");
  }
  const std::size_t nq = queries.size() / dim_;
  const std::size_t kk = std::min(k, n_vectors_);
  return search_landmark_rerank(queries, nq, dim_, kk, landmark_basis_q4_, scan_dim_, lm_centroids_,
                                scan_bits_, blocked_codes_, n_blocks_, scales_, residual_norms_,
                                rotation_, scale_resid_, timings, sign_codes_, sign_n_blocks_);
}

}  // namespace vectorcache
