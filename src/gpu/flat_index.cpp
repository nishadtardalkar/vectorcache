#include "vectorcache/gpu/flat_index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "kernels.cuh"
#include "vectorcache/constants.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache::gpu {
namespace {

std::size_t pad_scan_dim(std::size_t m) { return (m + 7u) & ~std::size_t{7u}; }

float l2_normalize_into(const float* src, float* dst, std::size_t dim) {
  double s = 0.0;
  for (std::size_t i = 0; i < dim; ++i) s += static_cast<double>(src[i]) * src[i];
  const float nrm = static_cast<float>(std::sqrt(s));
  if (nrm < kMinInputNorm) {
    std::fill(dst, dst + static_cast<std::ptrdiff_t>(dim), 0.f);
    return 0.f;
  }
  const float inv = 1.f / nrm;
  for (std::size_t d = 0; d < dim; ++d) dst[d] = src[d] * inv;
  return nrm;
}

std::size_t quantize_coord(float x, std::span<const float> boundaries) {
  std::size_t v = 0;
  for (float b : boundaries) v += static_cast<std::size_t>(x >= b);
  return v;
}

float norm_sq_f32(const float* x, std::size_t dim) {
  float s = 0.f;
  for (std::size_t i = 0; i < dim; ++i) s += x[i] * x[i];
  return s;
}

float encode_scaled_yhat(const float* src, float* unit_scratch, float* rot_scratch, float* yhat_out,
                         std::size_t dim, const Rotation& rotation,
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

struct FlatIndex::DeviceState {
  detail::CudaBuffers buf;
  Rotation rotation;
  explicit DeviceState(std::size_t dim) : rotation(dim) {}
};

bool cuda_available() { return detail::device_count() > 0; }

FlatIndex::FlatIndex(std::size_t dim, std::size_t bits) : FlatIndex(dim, bits, 0, 0, false) {}

FlatIndex::FlatIndex(std::size_t dim, std::size_t bits, std::size_t n_landmarks,
                     std::uint64_t landmark_seed, bool project_then_quant)
    : dim_(dim),
      bits_(bits),
      n_landmarks_(n_landmarks),
      scan_dim_(n_landmarks > 0 ? pad_scan_dim(n_landmarks) : dim),
      landmark_seed_(landmark_seed),
      project_then_quant_(project_then_quant) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("GpuFlatIndex dim must be a positive multiple of 8, <= 16384");
  }
  if (bits < 2 || bits > 4) {
    throw std::invalid_argument("GpuFlatIndex bits must be 2, 3, or 4");
  }
  if (n_landmarks > dim) {
    throw std::invalid_argument("n_landmarks must be <= dim");
  }
  if (n_landmarks > 0 && scan_dim_ % 8 != 0) {
    throw std::invalid_argument("scan_dim must be a multiple of 8");
  }
  if (!cuda_available()) {
    throw std::runtime_error("CUDA device not available");
  }
  char name[256];
  detail::device_name(name, sizeof(name));
  device_name_ = name;

  if (n_landmarks_ == 0) {
    auto cb = codebook(bits_, dim_);
    boundaries_ = std::move(cb.first);
    centroids_ = std::move(cb.second);
    lm_boundaries_ = boundaries_;
    lm_centroids_ = centroids_;
  } else {
    auto ambient = codebook(bits_, dim_);
    boundaries_ = std::move(ambient.first);
    centroids_ = std::move(ambient.second);
    auto lm = codebook(bits_, scan_dim_);
    lm_boundaries_ = std::move(lm.first);
    lm_centroids_ = std::move(lm.second);
    landmark_basis_ = make_ortho_landmarks(dim_, n_landmarks_, landmark_seed_);
    landmark_basis_q4_ = quantize_landmarks(landmark_basis_, dim_, n_landmarks_);
  }
  dev_ = std::make_unique<DeviceState>(dim_);
}

FlatIndex::~FlatIndex() = default;
FlatIndex::FlatIndex(FlatIndex&&) noexcept = default;
FlatIndex& FlatIndex::operator=(FlatIndex&&) noexcept = default;

void FlatIndex::add(std::span<const float> vectors) {
  if (vectors.empty()) return;
  if (vectors.size() % dim_ != 0) {
    throw std::invalid_argument("add: vectors size must be multiple of dim");
  }
  if (prepared_) {
    throw std::runtime_error("add after prepare is not supported for GpuFlatIndex");
  }
  const std::size_t n = vectors.size() / dim_;
  host_vectors_.insert(host_vectors_.end(), vectors.begin(), vectors.end());
  n_vectors_ += n;
}

void FlatIndex::prepare() {
  if (prepared_) return;
  if (n_vectors_ == 0) {
    throw std::runtime_error("GpuFlatIndex::prepare with empty index");
  }
  auto& b = dev_->buf;

  if (n_landmarks_ == 0) {
    detail::alloc_index_buffers(b, n_vectors_, dim_, scan_dim_, bits_, dev_->rotation.block(),
                                dev_->rotation.inv_sqrt_block(), 0);
    detail::upload_rotation(b, dev_->rotation.perm(0).data(), dev_->rotation.signs(0).data());
    detail::upload_codebook(b, centroids_.data(), boundaries_.data());
    detail::upload_vectors(b, host_vectors_.data(), n_vectors_);
    detail::encode_on_device(b);
  } else {
    // Host: ambient rotate/TQ → project onto L → residual norms (CPU LandmarkIndex parity).
    std::vector<float> landmark_rows(n_vectors_ * scan_dim_);
    residual_norms_.assign(n_vectors_, 0.f);
    constexpr std::size_t kTile = 64;
    const float* L = landmark_basis_.data();
    const auto boundaries = std::span<const float>(boundaries_);
    const auto centroids = std::span<const float>(centroids_);

    for (std::size_t tile = 0; tile < n_vectors_; tile += kTile) {
      const std::size_t tile_n = std::min(kTile, n_vectors_ - tile);
      std::vector<float> Y(tile_n * dim_);
      std::vector<float> v_norm2(tile_n);
      std::vector<float> unit_scratch(dim_);
      std::vector<float> rot_scratch(dim_);
      for (std::size_t i = 0; i < tile_n; ++i) {
        const float* src = host_vectors_.data() + (tile + i) * dim_;
        float* yrow = Y.data() + i * dim_;
        if (project_then_quant_) {
          const float nrm = l2_normalize_into(src, unit_scratch.data(), dim_);
          if (nrm < kMinInputNorm) {
            std::fill(yrow, yrow + static_cast<std::ptrdiff_t>(dim_), 0.f);
            v_norm2[i] = 0.f;
          } else {
            std::copy(unit_scratch.begin(), unit_scratch.end(), rot_scratch.begin());
            dev_->rotation.apply_with_scratch(std::span<float>(rot_scratch.data(), dim_),
                                              std::span<float>(unit_scratch.data(), dim_));
            std::copy(rot_scratch.begin(), rot_scratch.end(), yrow);
            v_norm2[i] = 1.f;
          }
        } else {
          v_norm2[i] =
              encode_scaled_yhat(src, unit_scratch.data(), rot_scratch.data(), yrow, dim_,
                                 dev_->rotation, boundaries, centroids);
        }
      }
      float* lm_out = landmark_rows.data() + tile * scan_dim_;
      project_rows_onto_landmarks(Y.data(), tile_n, dim_, L, n_landmarks_, lm_out, scan_dim_);
      for (std::size_t i = 0; i < tile_n; ++i) {
        const float land_norm2 = norm_sq_f32(lm_out + i * scan_dim_, n_landmarks_);
        residual_norms_[tile + i] = std::sqrt(std::max(0.f, v_norm2[i] - land_norm2));
      }
    }

    detail::alloc_index_buffers(b, n_vectors_, dim_, scan_dim_, bits_, dev_->rotation.block(),
                                dev_->rotation.inv_sqrt_block(), n_landmarks_);
    detail::upload_rotation(b, dev_->rotation.perm(0).data(), dev_->rotation.signs(0).data());
    detail::upload_codebook(b, lm_centroids_.data(), lm_boundaries_.data());
    detail::upload_vectors(b, landmark_rows.data(), n_vectors_);
    detail::upload_residuals(b, residual_norms_.data(), n_vectors_);
    detail::upload_landmark_basis(b, landmark_basis_.data(), dim_, n_landmarks_);
    detail::encode_prerotated_on_device(b);
  }

  host_vectors_.clear();
  host_vectors_.shrink_to_fit();
  prepared_ = true;
}

SearchResults FlatIndex::search(std::span<const float> queries, std::size_t k) const {
  if (!prepared_) {
    throw std::runtime_error("GpuFlatIndex::search before prepare");
  }
  if (queries.empty()) {
    SearchResults empty;
    empty.nq = 0;
    empty.k = k;
    return empty;
  }
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("search: queries size must be multiple of dim");
  }
  const std::size_t nq = queries.size() / dim_;
  const std::size_t kk = std::min(k, n_vectors_);
  SearchResults out;
  out.nq = nq;
  out.k = kk;
  out.scores.resize_uninitialized(nq * kk);
  out.ids.resize_uninitialized(nq * kk);
  detail::search_on_device(dev_->buf, queries.data(), nq, kk, out.scores.data(), out.ids.data());
  return out;
}

}  // namespace vectorcache::gpu
