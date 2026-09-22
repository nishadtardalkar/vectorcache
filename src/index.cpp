#include "vectorcache/index.hpp"

#include <stdexcept>

#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"

namespace vectorcache {

TurboQuantIndex::TurboQuantIndex(std::size_t dim, std::size_t bit_width)
    : dim_(dim), bits_(bit_width), rotation_(dim) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("dim must be a multiple of 8 in [8, MAX_DIM]");
  }
  if (bit_width < 2 || bit_width > 4) {
    throw std::invalid_argument("bit_width must be 2, 3, or 4");
  }
  auto cb = codebook(bits_, dim_);
  boundaries_ = std::move(cb.first);
  centroids_ = std::move(cb.second);
}

void TurboQuantIndex::add(std::span<const float> vectors) {
  if (vectors.size() % dim_ != 0) {
    throw std::invalid_argument("vectors length must be multiple of dim");
  }
  const std::size_t n = vectors.size() / dim_;
  if (n == 0) {
    return;
  }
  const Calibration* cal = calibration_ ? &*calibration_ : nullptr;
  encode(vectors, n, dim_, rotation_, boundaries_, centroids_, bits_, cal, rotated_scratch_,
         packed_, scales_);
  blocked_ready_ = false;
}

void TurboQuantIndex::calibrate(std::span<const float> sample) {
  if (sample.size() % dim_ != 0) {
    throw std::invalid_argument("sample length must be multiple of dim");
  }
  const std::size_t n = sample.size() / dim_;
  if (n < kMinCalibrationRows) {
    throw std::invalid_argument("need at least 2 calibration rows");
  }
  calibration_ = fit_calibration(sample, n, dim_, rotation_, centroids_, rotated_scratch_);
  // Re-encode existing rows if any: reconstruct not available without float originals.
  // For in-RAM bench workflow, calibrate before add.
  if (!scales_.empty()) {
    throw std::runtime_error("calibrate after add is not supported; call calibrate before add");
  }
}

void TurboQuantIndex::prepare() { ensure_blocked(); }

void TurboQuantIndex::ensure_blocked() const {
  if (blocked_ready_) {
    return;
  }
  auto [blocked, n_blocks] = repack(packed_, scales_.size(), bits_, dim_);
  blocked_ = std::move(blocked);
  n_blocks_ = n_blocks;
  blocked_ready_ = true;
}

SearchResults TurboQuantIndex::search(std::span<const float> queries, std::size_t k) const {
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("queries length must be multiple of dim");
  }
  const std::size_t nq = queries.size() / dim_;
  ensure_blocked();
  std::span<const float> shift;
  std::span<const float> scale;
  if (calibration_) {
    shift = calibration_->shift;
    scale = calibration_->scale_tq;
  }
  return search_flat(queries, nq, dim_, k, rotation_, centroids_, bits_, blocked_, n_blocks_,
                     scales_, shift, scale);
}

}  // namespace vectorcache
