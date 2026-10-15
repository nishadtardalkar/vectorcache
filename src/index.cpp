#include "vectorcache/index.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/search/landmark_scan.hpp"
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
  // boundaries has (n_levels-1) midpoints, sorted ascending.
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

}  // namespace

LandmarkIndex::LandmarkIndex(std::size_t dim, std::size_t bit_width, std::size_t n_landmarks,
                             std::size_t shortlist, std::uint64_t landmark_seed)
    : dim_(dim),
      bits_(bit_width),
      n_landmarks_(n_landmarks),
      shortlist_(shortlist),
      landmark_seed_(landmark_seed),
      rotation_(dim) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("dim must be a multiple of 8 in [8, MAX_DIM]");
  }
  if (bit_width < 2 || bit_width > 4) {
    throw std::invalid_argument("bit_width must be 2, 3, or 4");
  }
  if (n_landmarks == 0 || n_landmarks > dim) {
    throw std::invalid_argument("n_landmarks must be in 1..dim");
  }
  if (shortlist == 0) {
    throw std::invalid_argument("shortlist must be > 0");
  }
  auto cb = codebook(bits_, dim_);
  boundaries_ = std::move(cb.first);
  centroids_ = std::move(cb.second);
  landmark_basis_ = make_ortho_landmarks(dim_, n_landmarks_, landmark_seed_);
  rotate_scratch_.resize(dim_);
}

void LandmarkIndex::encode_row(const float* src, float* unit_out, float* landmark_row_out) {
  l2_normalize_into(src, unit_out, dim_);
  std::copy(unit_out, unit_out + static_cast<std::ptrdiff_t>(dim_), rotate_scratch_.begin());
  rotation_.apply(rotate_scratch_);

  // 4-bit scalar quantize + RaBitQ alpha in rotated space.
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

  // landmark_row = (alpha * yhat) @ L , L column-major dim x m
  for (std::size_t j = 0; j < n_landmarks_; ++j) {
    double s = 0.0;
    const float* Lj = landmark_basis_.data() + j * dim_;
    for (std::size_t i = 0; i < dim_; ++i) {
      s += static_cast<double>(alpha * yhat[i]) * Lj[i];
    }
    landmark_row_out[j] = static_cast<float>(s);
  }
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
  landmark_rows_.resize(n_vectors_ * n_landmarks_);
  prepared_ = false;

  for (std::size_t i = 0; i < n; ++i) {
    encode_row(vectors.data() + i * dim_, db_unit_.data() + (base + i) * dim_,
               landmark_rows_.data() + (base + i) * n_landmarks_);
  }
}

void LandmarkIndex::prepare() {
  n_blocks_ = (n_vectors_ + kBlock - 1) / kBlock;
  blocked_landmarks_ = pack_landmark_blocks(landmark_rows_, n_vectors_, n_landmarks_);
  prepared_ = true;
}

SearchResults LandmarkIndex::search(std::span<const float> queries, std::size_t k) const {
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("queries length must be multiple of dim");
  }
  if (!prepared_) {
    throw std::runtime_error("call prepare() before search");
  }
  const std::size_t nq = queries.size() / dim_;
  const std::size_t sl = std::min(shortlist_, n_vectors_);
  const std::size_t kk = std::min(k, n_vectors_);
  return search_landmark_rerank(queries, nq, dim_, kk, sl, landmark_basis_, n_landmarks_,
                                blocked_landmarks_, n_vectors_, n_blocks_, db_unit_, rotation_);
}

}  // namespace vectorcache
