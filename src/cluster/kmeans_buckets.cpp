#include "vectorcache/cluster/kmeans_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "vectorcache/cluster/score_centroids.hpp"
#include "vectorcache/constants.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"

namespace vectorcache {
namespace {

float dot_row(const float* a, const float* b, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    s += static_cast<double>(a[d]) * static_cast<double>(b[d]);
  }
  return static_cast<float>(s);
}

void copy_normalize_row(std::span<const float> src, float* dest, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    dest[d] = src[d];
    s += static_cast<double>(src[d]) * static_cast<double>(src[d]);
  }
  const float inv = (s > static_cast<double>(kMinInputNorm) * kMinInputNorm)
                        ? static_cast<float>(1.0 / std::sqrt(s))
                        : 0.f;
  for (std::size_t d = 0; d < dim; ++d) dest[d] *= inv;
}

void normalize_from_sum(const double* sum, float* out, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) s += sum[d] * sum[d];
  const double inv = (s > static_cast<double>(kMinInputNorm) * kMinInputNorm) ? (1.0 / std::sqrt(s))
                                                                             : 0.0;
  for (std::size_t d = 0; d < dim; ++d) out[d] = static_cast<float>(sum[d] * inv);
}

void normalize_inplace(float* v, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) s += static_cast<double>(v[d]) * static_cast<double>(v[d]);
  const float inv = (s > static_cast<double>(kMinInputNorm) * kMinInputNorm)
                        ? static_cast<float>(1.0 / std::sqrt(s))
                        : 0.f;
  for (std::size_t d = 0; d < dim; ++d) v[d] *= inv;
}

}  // namespace

BucketedTurboQuantIndex::BucketedTurboQuantIndex(std::size_t dim, std::size_t bit_width,
                                                 BucketParams params)
    : dim_(dim), bits_(bit_width), params_(params), rotation_(dim) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("dim must be a multiple of 8 in [8, MAX_DIM]");
  }
  if (bit_width < 2 || bit_width > 4) {
    throw std::invalid_argument("bit_width must be 2, 3, or 4");
  }
  if (!(params_.scan_fraction > 0.f && params_.scan_fraction <= 1.f)) {
    throw std::invalid_argument("scan_fraction must be in (0, 1]");
  }
  if (params_.max_bucket_size < 2) {
    throw std::invalid_argument("max_bucket_size must be >= 2");
  }
  if (params_.energy_soft < 2) {
    throw std::invalid_argument("energy_soft must be >= 2");
  }
  auto cb = codebook(bits_, dim_);
  boundaries_ = std::move(cb.first);
  codebook_centroids_ = std::move(cb.second);
}

void BucketedTurboQuantIndex::calibrate(std::span<const float> sample) {
  if (sample.size() % dim_ != 0) {
    throw std::invalid_argument("sample length must be multiple of dim");
  }
  const std::size_t n = sample.size() / dim_;
  if (n < kMinCalibrationRows) {
    throw std::invalid_argument("need at least 2 calibration rows");
  }
  if (next_id_ != 0) {
    throw std::runtime_error("calibrate after add is not supported; call calibrate before add");
  }
  calibration_ = fit_calibration(sample, n, dim_, rotation_, codebook_centroids_, rotated_scratch_);
}

void BucketedTurboQuantIndex::sync_centroid_matrices() {
  routing_centroid_matrix_.resize(buckets_.size() * dim_);
  encode_centroid_matrix_.resize(buckets_.size() * dim_);
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    sync_routing_row(i);
    sync_encode_row(i);
  }
}

void BucketedTurboQuantIndex::sync_routing_row(std::size_t bi) {
  std::copy(buckets_[bi].routing_centroid.begin(), buckets_[bi].routing_centroid.end(),
            routing_centroid_matrix_.begin() + static_cast<std::ptrdiff_t>(bi * dim_));
}

void BucketedTurboQuantIndex::sync_encode_row(std::size_t bi) {
  std::copy(buckets_[bi].encode_centroid.begin(), buckets_[bi].encode_centroid.end(),
            encode_centroid_matrix_.begin() + static_cast<std::ptrdiff_t>(bi * dim_));
}

std::size_t BucketedTurboQuantIndex::bucket_size(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_size");
  return buckets_[bucket].count;
}

std::span<const float> BucketedTurboQuantIndex::bucket_centroid(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_centroid");
  return buckets_[bucket].routing_centroid;
}

std::span<const float> BucketedTurboQuantIndex::bucket_encode_centroid(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_encode_centroid");
  return buckets_[bucket].encode_centroid;
}

double BucketedTurboQuantIndex::bucket_cos_variance(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_cos_variance");
  const Bucket& b = buckets_[bucket];
  if (b.count < 2) return 0.0;
  return b.cos_m2 / static_cast<double>(b.count - 1);
}

std::size_t BucketedTurboQuantIndex::spawn_bucket(std::span<const float> unit_row) {
  Bucket b;
  // Freeze encode at founding vector; append_member fills weighted sum / routing / count.
  b.encode_centroid.assign(unit_row.begin(), unit_row.end());
  b.routing_centroid.assign(unit_row.begin(), unit_row.end());
  b.sum.assign(dim_, 0.0);
  b.wsum = 0.0;
  b.count = 0;
  b.cos_mean = 0.0;
  b.cos_m2 = 0.0;
  buckets_.push_back(std::move(b));
  sync_centroid_matrices();
  return buckets_.size() - 1;
}

void BucketedTurboQuantIndex::soft_pull(std::size_t bi, std::span<const float> unit_row) {
  Bucket& b = buckets_[bi];
  if (b.count == 0) return;
  const double damp = static_cast<double>(params_.attract) / (1.0 + static_cast<double>(b.count) / 220.0);
  for (std::size_t d = 0; d < dim_; ++d) {
    b.routing_centroid[d] = static_cast<float>(static_cast<double>(b.routing_centroid[d]) +
                                               damp * static_cast<double>(unit_row[d]));
  }
  normalize_inplace(b.routing_centroid.data(), dim_);
  sync_routing_row(bi);
}

int BucketedTurboQuantIndex::nearest_rival(std::size_t bi) const {
  if (buckets_.size() < 2) return -1;
  const float* ci = routing_centroid_matrix_.data() + bi * dim_;
  float best = -std::numeric_limits<float>::infinity();
  int best_j = -1;
  for (std::size_t j = 0; j < buckets_.size(); ++j) {
    if (j == bi || buckets_[j].count == 0) continue;
    const float ip = dot_row(ci, routing_centroid_matrix_.data() + j * dim_, dim_);
    if (ip > best) {
      best = ip;
      best_j = static_cast<int>(j);
    }
  }
  return best_j;
}

void BucketedTurboQuantIndex::kill_bucket(std::size_t bi) {
  const std::size_t last = buckets_.size() - 1;
  if (bi != last) {
    buckets_[bi] = std::move(buckets_[last]);
  }
  buckets_.pop_back();
  sync_centroid_matrices();
}

void BucketedTurboQuantIndex::absorb_into(std::size_t keep, std::size_t drop) {
  if (keep == drop) return;
  Bucket& a = buckets_[keep];
  Bucket& b = buckets_[drop];
  const std::size_t na = a.count;
  const std::size_t nb = b.count;
  a.floats.resize((na + nb) * dim_);
  std::copy(b.floats.begin(), b.floats.end(),
            a.floats.begin() + static_cast<std::ptrdiff_t>(na * dim_));
  a.ids.insert(a.ids.end(), b.ids.begin(), b.ids.end());
  std::vector<float> units = std::move(a.floats);
  std::vector<std::uint64_t> ids = std::move(a.ids);
  finalize_partition(a, std::move(units), std::move(ids));
  sync_routing_row(keep);
  sync_encode_row(keep);
  kill_bucket(drop);
}

bool BucketedTurboQuantIndex::maybe_absorb(std::size_t bi) {
  if (bi >= buckets_.size()) return false;
  const std::size_t n = buckets_[bi].count;
  if (n == 0 || n > params_.absorb_max) return false;
  const int rj = nearest_rival(bi);
  if (rj < 0) return false;
  const float ip = dot_row(routing_centroid_matrix_.data() + bi * dim_,
                           routing_centroid_matrix_.data() + static_cast<std::size_t>(rj) * dim_,
                           dim_);
  if (ip < params_.absorb_ip) return false;
  const std::size_t keep = bi < static_cast<std::size_t>(rj) ? bi : static_cast<std::size_t>(rj);
  const std::size_t drop = bi < static_cast<std::size_t>(rj) ? static_cast<std::size_t>(rj) : bi;
  absorb_into(keep, drop);
  return true;
}

void BucketedTurboQuantIndex::encode_units_into(Bucket& b) {
  b.packed.clear();
  b.scales.clear();
  b.blocked.clear();
  b.n_blocks = 0;
  b.blocked_ready = false;
  if (b.count == 0) return;

  std::vector<float> residual(b.count * dim_);
  for (std::size_t i = 0; i < b.count; ++i) {
    const float* unit = b.floats.data() + i * dim_;
    float* r = residual.data() + i * dim_;
    for (std::size_t d = 0; d < dim_; ++d) {
      r[d] = unit[d] - b.encode_centroid[d];
    }
  }
  const Calibration* cal = calibration_ ? &*calibration_ : nullptr;
  encode(residual, b.count, dim_, rotation_, boundaries_, codebook_centroids_, bits_, cal,
         rotated_scratch_, b.packed, b.scales);
}

void BucketedTurboQuantIndex::finalize_partition(Bucket& b, std::vector<float> units,
                                                 std::vector<std::uint64_t> ids) {
  const std::size_t n = ids.size();
  b.ids = std::move(ids);
  b.floats = std::move(units);
  b.count = n;
  b.sum.assign(dim_, 0.0);
  b.wsum = static_cast<double>(n);
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = b.floats.data() + i * dim_;
    for (std::size_t d = 0; d < dim_; ++d) b.sum[d] += static_cast<double>(row[d]);
  }
  b.encode_centroid.resize(dim_);
  b.routing_centroid.resize(dim_);
  if (n == 0) {
    std::fill(b.encode_centroid.begin(), b.encode_centroid.end(), 0.f);
    std::fill(b.routing_centroid.begin(), b.routing_centroid.end(), 0.f);
    b.wsum = 0.0;
    b.cos_mean = 0.0;
    b.cos_m2 = 0.0;
    encode_units_into(b);
    return;
  }
  normalize_from_sum(b.sum.data(), b.encode_centroid.data(), dim_);
  b.routing_centroid = b.encode_centroid;

  b.cos_mean = 0.0;
  b.cos_m2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float cos = dot_row(b.floats.data() + i * dim_, b.routing_centroid.data(), dim_);
    const double c = static_cast<double>(cos);
    const double k = static_cast<double>(i + 1);
    const double delta = c - b.cos_mean;
    b.cos_mean += delta / k;
    b.cos_m2 += delta * (c - b.cos_mean);
  }

  encode_units_into(b);
}

void BucketedTurboQuantIndex::split_2means(std::size_t bi) {
  Bucket& parent = buckets_[bi];
  const std::size_t n = parent.count;
  if (n < 4) return;

  // 1-iter spherical 2-means poles, then Voronoi membership (champ 2means_part).
  std::vector<double> p0(dim_), p1(dim_);
  std::size_t i_lo = 0;
  float lo_cos = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const float cos = dot_row(parent.floats.data() + i * dim_, parent.routing_centroid.data(), dim_);
    if (cos < lo_cos) {
      lo_cos = cos;
      i_lo = i;
    }
  }
  for (std::size_t d = 0; d < dim_; ++d) p0[d] = parent.floats[i_lo * dim_ + d];
  {
    double s = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) s += p0[d] * p0[d];
    const double inv = s > 0.0 ? 1.0 / std::sqrt(s) : 0.0;
    for (std::size_t d = 0; d < dim_; ++d) p0[d] *= inv;
  }
  std::size_t i_hi = 0;
  float hi_cos = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    double ip = 0.0;
    const float* row = parent.floats.data() + i * dim_;
    for (std::size_t d = 0; d < dim_; ++d) ip += static_cast<double>(row[d]) * p0[d];
    if (static_cast<float>(ip) < hi_cos) {
      hi_cos = static_cast<float>(ip);
      i_hi = i;
    }
  }
  for (std::size_t d = 0; d < dim_; ++d) p1[d] = parent.floats[i_hi * dim_ + d];
  {
    double s = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) s += p1[d] * p1[d];
    const double inv = s > 0.0 ? 1.0 / std::sqrt(s) : 0.0;
    for (std::size_t d = 0; d < dim_; ++d) p1[d] *= inv;
  }

  for (int iter = 0; iter < 1; ++iter) {
    std::vector<double> q0(dim_, 0.0), q1(dim_, 0.0);
    std::size_t n0 = 0, n1 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float* row = parent.floats.data() + i * dim_;
      double s0 = 0.0, s1 = 0.0;
      for (std::size_t d = 0; d < dim_; ++d) {
        s0 += static_cast<double>(row[d]) * p0[d];
        s1 += static_cast<double>(row[d]) * p1[d];
      }
      if (s0 >= s1) {
        for (std::size_t d = 0; d < dim_; ++d) q0[d] += static_cast<double>(row[d]);
        ++n0;
      } else {
        for (std::size_t d = 0; d < dim_; ++d) q1[d] += static_cast<double>(row[d]);
        ++n1;
      }
    }
    if (n0 > 0) {
      double s = 0.0;
      for (std::size_t d = 0; d < dim_; ++d) s += q0[d] * q0[d];
      const double inv = s > 0.0 ? 1.0 / std::sqrt(s) : 0.0;
      for (std::size_t d = 0; d < dim_; ++d) p0[d] = q0[d] * inv;
    }
    if (n1 > 0) {
      double s = 0.0;
      for (std::size_t d = 0; d < dim_; ++d) s += q1[d] * q1[d];
      const double inv = s > 0.0 ? 1.0 / std::sqrt(s) : 0.0;
      for (std::size_t d = 0; d < dim_; ++d) p1[d] = q1[d] * inv;
    }
  }

  std::vector<std::uint8_t> side(n, 0);
  std::size_t n_a = 0, n_b = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = parent.floats.data() + i * dim_;
    double s0 = 0.0, s1 = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) {
      s0 += static_cast<double>(row[d]) * p0[d];
      s1 += static_cast<double>(row[d]) * p1[d];
    }
    if (s0 >= s1) {
      side[i] = 0;
      ++n_a;
    } else {
      side[i] = 1;
      ++n_b;
    }
  }

  if (n_a < 2 || n_b < 2) {
    // Median fallback on (p1-p0) axis through centroid.
    std::vector<double> axis(dim_);
    double an = 0.0;
    for (std::size_t d = 0; d < dim_; ++d) {
      axis[d] = p1[d] - p0[d];
      an += axis[d] * axis[d];
    }
    const double inv = an > 0.0 ? 1.0 / std::sqrt(an) : 0.0;
    for (std::size_t d = 0; d < dim_; ++d) axis[d] *= inv;
    std::vector<std::pair<float, std::size_t>> order(n);
    for (std::size_t i = 0; i < n; ++i) {
      const float* row = parent.floats.data() + i * dim_;
      double sc = 0.0;
      for (std::size_t d = 0; d < dim_; ++d) {
        sc += (static_cast<double>(row[d]) - static_cast<double>(parent.routing_centroid[d])) *
              axis[d];
      }
      order[i] = {static_cast<float>(sc), i};
    }
    std::sort(order.begin(), order.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    std::fill(side.begin(), side.end(), static_cast<std::uint8_t>(0));
    n_a = 0;
    n_b = 0;
    const std::size_t mid = n / 2;
    for (std::size_t r = 0; r < mid; ++r) {
      side[order[r].second] = 0;
      ++n_a;
    }
    for (std::size_t r = mid; r < n; ++r) {
      side[order[r].second] = 1;
      ++n_b;
    }
  }

  std::vector<float> units_a(n_a * dim_), units_b(n_b * dim_);
  std::vector<std::uint64_t> ids_a(n_a), ids_b(n_b);
  std::size_t ia = 0, ib = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = parent.floats.data() + i * dim_;
    if (side[i] == 0) {
      std::copy(row, row + dim_, units_a.begin() + static_cast<std::ptrdiff_t>(ia * dim_));
      ids_a[ia++] = parent.ids[i];
    } else {
      std::copy(row, row + dim_, units_b.begin() + static_cast<std::ptrdiff_t>(ib * dim_));
      ids_b[ib++] = parent.ids[i];
    }
  }

  finalize_partition(parent, std::move(units_a), std::move(ids_a));
  sync_routing_row(bi);
  sync_encode_row(bi);

  Bucket child;
  finalize_partition(child, std::move(units_b), std::move(ids_b));
  buckets_.push_back(std::move(child));
  sync_centroid_matrices();
}

std::size_t BucketedTurboQuantIndex::soft_threshold(std::size_t stream_i, std::size_t count) const {
  if (params_.expected_n == 0) return params_.energy_soft;
  const auto pulse_start =
      static_cast<std::size_t>(params_.pulse_start_frac * static_cast<double>(params_.expected_n));
  const auto pulse_end =
      static_cast<std::size_t>(params_.pulse_end_frac * static_cast<double>(params_.expected_n));
  std::size_t soft_t;
  if (stream_i >= pulse_start && stream_i < pulse_end) {
    soft_t = params_.pulse_soft;
  } else if (stream_i >= pulse_end) {
    const double t = static_cast<double>(stream_i - pulse_end) /
                     std::max<double>(static_cast<double>(params_.expected_n - pulse_end), 1.0);
    soft_t = params_.ramp_soft0 +
             static_cast<std::size_t>(t * static_cast<double>(params_.ramp_soft1 - params_.ramp_soft0));
    if (count >= params_.fat_soft_n) soft_t += params_.fat_soft_extra;
  } else {
    soft_t = params_.energy_soft;
  }
  return soft_t;
}

float BucketedTurboQuantIndex::soft_trig(std::size_t stream_i) const {
  if (params_.expected_n == 0) return params_.energy_trig;
  const auto pulse_start =
      static_cast<std::size_t>(params_.pulse_start_frac * static_cast<double>(params_.expected_n));
  const auto pulse_end =
      static_cast<std::size_t>(params_.pulse_end_frac * static_cast<double>(params_.expected_n));
  if (stream_i >= pulse_start && stream_i < pulse_end) return params_.pulse_trig;
  return params_.energy_trig;
}

void BucketedTurboQuantIndex::append_member(std::size_t bi, std::span<const float> unit_row,
                                            std::uint64_t id, float margin) {
  Bucket& b = buckets_[bi];

  const float cos = (b.count == 0) ? 1.f : dot_row(unit_row.data(), b.routing_centroid.data(), dim_);
  {
    const double c = static_cast<double>(cos);
    const double n_new = static_cast<double>(b.count + 1);
    const double delta = c - b.cos_mean;
    b.cos_mean += delta / n_new;
    b.cos_m2 += delta * (c - b.cos_mean);
  }

  const float w = std::clamp(margin / (margin + params_.margin_tau), 0.1f, 1.0f);
  if (b.sum.size() != dim_) b.sum.assign(dim_, 0.0);
  for (std::size_t d = 0; d < dim_; ++d) {
    b.sum[d] += static_cast<double>(w) * static_cast<double>(unit_row[d]);
  }
  b.wsum += static_cast<double>(w);
  ++b.count;
  b.routing_centroid.resize(dim_);
  if (b.wsum > 0.0) {
    for (std::size_t d = 0; d < dim_; ++d) {
      b.routing_centroid[d] = static_cast<float>(b.sum[d] / b.wsum);
    }
    normalize_inplace(b.routing_centroid.data(), dim_);
  }
  sync_routing_row(bi);

  const std::size_t old_n = b.count - 1;
  b.floats.resize(b.count * dim_);
  std::copy(unit_row.begin(), unit_row.end(),
            b.floats.begin() + static_cast<std::ptrdiff_t>(old_n * dim_));
  b.ids.push_back(id);

  std::vector<float> residual(dim_);
  for (std::size_t d = 0; d < dim_; ++d) {
    residual[d] = unit_row[d] - b.encode_centroid[d];
  }
  const Calibration* cal = calibration_ ? &*calibration_ : nullptr;
  encode(residual, 1, dim_, rotation_, boundaries_, codebook_centroids_, bits_, cal,
         rotated_scratch_, b.packed, b.scales);
  b.blocked_ready = false;
}

void BucketedTurboQuantIndex::add(std::span<const float> vectors) {
  if (prepared_) {
    throw std::runtime_error("add after prepare is not supported for bucketed index");
  }
  if (vectors.size() % dim_ != 0) {
    throw std::invalid_argument("vectors length must be multiple of dim");
  }
  const std::size_t n = vectors.size() / dim_;
  if (n == 0) return;

  std::vector<float> row(dim_);

  for (std::size_t i = 0; i < n; ++i) {
    copy_normalize_row(vectors.subspan(i * dim_, dim_), row.data(), dim_);
    const std::size_t stream_i = stream_i_++;

    float margin = 1.f;
    std::size_t best_j = 0;
    int second_j = -1;
    if (buckets_.empty()) {
      best_j = spawn_bucket(std::span<const float>(row.data(), dim_));
    } else {
      float best_cos = -std::numeric_limits<float>::infinity();
      float second_cos = -std::numeric_limits<float>::infinity();
      const std::size_t nc = buckets_.size();
      bool have_best = false;
      for (std::size_t j = 0; j < nc; ++j) {
        if (buckets_[j].count == 0) continue;
        const float cos = dot_row(row.data(), routing_centroid_matrix_.data() + j * dim_, dim_);
        if (!have_best || cos > best_cos) {
          if (have_best) {
            second_cos = best_cos;
            second_j = static_cast<int>(best_j);
          }
          best_cos = cos;
          best_j = j;
          have_best = true;
        } else if (second_j < 0 || cos > second_cos) {
          second_cos = cos;
          second_j = static_cast<int>(j);
        }
      }
      if (second_j >= 0 && second_j != static_cast<int>(best_j) &&
          std::isfinite(second_cos)) {
        margin = std::max(0.f, best_cos - second_cos);
        if (margin < params_.attract_margin) {
          soft_pull(best_j, std::span<const float>(row.data(), dim_));
          soft_pull(static_cast<std::size_t>(second_j), std::span<const float>(row.data(), dim_));
        }
      }
    }

    const std::uint64_t id = next_id_++;
    // Champ spawn seeds S/wsum then appends the founding vector again.
    append_member(best_j, std::span<const float>(row.data(), dim_), id, margin);

    // Indices may change after absorb/split — re-resolve by id is unnecessary;
    // operate on best_j carefully, abort policy path if bucket moved.
    if (best_j >= buckets_.size()) continue;
    const std::size_t c_count = buckets_[best_j].count;
    const double energy = 1.0 - buckets_[best_j].cos_mean;

    if (c_count >= params_.max_bucket_size) {
      split_2means(best_j);
    } else {
      const std::size_t soft_t = soft_threshold(stream_i, c_count);
      const float trig = soft_trig(stream_i);
      const auto pulse_end =
          params_.expected_n == 0
              ? std::numeric_limits<std::size_t>::max()
              : static_cast<std::size_t>(params_.pulse_end_frac *
                                        static_cast<double>(params_.expected_n));
      const auto late_start =
          params_.expected_n == 0
              ? std::numeric_limits<std::size_t>::max()
              : static_cast<std::size_t>(params_.late_absorb_frac *
                                        static_cast<double>(params_.expected_n));

      if (c_count >= soft_t && energy >= static_cast<double>(trig)) {
        split_2means(best_j);
      } else if (stream_i >= pulse_end && c_count >= 8 && c_count <= params_.recov_absorb_max &&
                 (stream_i % 11 == 0)) {
        const int rj = nearest_rival(best_j);
        if (rj >= 0 && buckets_[static_cast<std::size_t>(rj)].count >= params_.recov_well_min) {
          const float ip =
              dot_row(routing_centroid_matrix_.data() + best_j * dim_,
                      routing_centroid_matrix_.data() + static_cast<std::size_t>(rj) * dim_, dim_);
          if (ip >= params_.recov_absorb_ip) {
            const std::size_t rj_u = static_cast<std::size_t>(rj);
            const std::size_t keep =
                buckets_[rj_u].count >= buckets_[best_j].count ? rj_u : best_j;
            const std::size_t drop = keep == rj_u ? best_j : rj_u;
            if (buckets_[drop].count <= params_.recov_absorb_max) {
              absorb_into(keep, drop);
            }
          }
        }
      } else if (stream_i >= late_start && c_count >= 8 && c_count <= params_.late_absorb_max &&
                 (stream_i % 13 == 0)) {
        const int rj = nearest_rival(best_j);
        if (rj >= 0 && buckets_[static_cast<std::size_t>(rj)].count >= params_.late_well_min) {
          const float ip =
              dot_row(routing_centroid_matrix_.data() + best_j * dim_,
                      routing_centroid_matrix_.data() + static_cast<std::size_t>(rj) * dim_, dim_);
          if (ip >= params_.late_absorb_ip) {
            const std::size_t rj_u = static_cast<std::size_t>(rj);
            const std::size_t keep =
                buckets_[rj_u].count >= buckets_[best_j].count ? rj_u : best_j;
            const std::size_t drop = keep == rj_u ? best_j : rj_u;
            if (buckets_[drop].count <= params_.late_absorb_max) {
              absorb_into(keep, drop);
            }
          }
        }
      } else if (c_count >= 8 && c_count <= params_.absorb_max && (stream_i % 17 == 0)) {
        maybe_absorb(best_j);
      }
    }
  }
}

void BucketedTurboQuantIndex::bake_rival_asym() {
  const std::size_t nc = buckets_.size();
  if (nc < 2) return;
  std::vector<float> baked = routing_centroid_matrix_;
  for (std::size_t bi = 0; bi < nc; ++bi) {
    if (buckets_[bi].count == 0) continue;
    const int rj = nearest_rival(bi);
    if (rj < 0) continue;
    const std::size_t nb = std::max<std::size_t>(buckets_[bi].count, 1);
    const std::size_t nr = std::max<std::size_t>(buckets_[static_cast<std::size_t>(rj)].count, 1);
    const double beta =
        static_cast<double>(params_.rival_push) *
        (1.0 + 0.5 * std::max(0.0, (static_cast<double>(nr) - static_cast<double>(nb)) /
                                       static_cast<double>(nr)));
    float* out = baked.data() + bi * dim_;
    const float* ci = routing_centroid_matrix_.data() + bi * dim_;
    const float* cr = routing_centroid_matrix_.data() + static_cast<std::size_t>(rj) * dim_;
    for (std::size_t d = 0; d < dim_; ++d) {
      out[d] = static_cast<float>(static_cast<double>(ci[d]) - beta * static_cast<double>(cr[d]));
    }
    normalize_inplace(out, dim_);
  }
  for (std::size_t bi = 0; bi < nc; ++bi) {
    if (buckets_[bi].count == 0) continue;
    std::copy(baked.begin() + static_cast<std::ptrdiff_t>(bi * dim_),
              baked.begin() + static_cast<std::ptrdiff_t>((bi + 1) * dim_),
              buckets_[bi].routing_centroid.begin());
    sync_routing_row(bi);
  }
}

void BucketedTurboQuantIndex::ensure_bucket_blocked(std::size_t bi) const {
  const Bucket& b = buckets_[bi];
  if (b.blocked_ready) return;
  if (b.count == 0) {
    b.blocked.clear();
    b.n_blocks = 0;
    b.blocked_ready = true;
    return;
  }
  auto [blocked, n_blocks] = repack(b.packed, b.scales.size(), bits_, dim_);
  b.blocked = std::move(blocked);
  b.n_blocks = n_blocks;
  b.blocked_ready = true;
}

void BucketedTurboQuantIndex::prepare() {
  bake_rival_asym();
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    ensure_bucket_blocked(i);
    Bucket& b = buckets_[i];
    b.floats.clear();
    b.floats.shrink_to_fit();
    b.packed.clear();
    b.packed.shrink_to_fit();
    b.sum.clear();
    b.sum.shrink_to_fit();
  }
  prepared_ = true;
}

SearchResults BucketedTurboQuantIndex::search(std::span<const float> queries, std::size_t k) const {
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("queries length must be multiple of dim");
  }
  const std::size_t nq = queries.size() / dim_;
  if (nq == 0 || next_id_ == 0 || k == 0) {
    SearchResults empty;
    empty.nq = nq;
    empty.k = 0;
    return empty;
  }

  for (std::size_t i = 0; i < buckets_.size(); ++i) ensure_bucket_blocked(i);

  std::span<const float> shift;
  std::span<const float> scale;
  if (calibration_) {
    shift = calibration_->shift;
    scale = calibration_->scale_tq;
  }
  auto prep =
      prepare_queries(queries, nq, dim_, rotation_, codebook_centroids_, bits_, shift, scale);

  std::vector<float> q_unit(nq * dim_);
  std::copy(queries.begin(), queries.end(), q_unit.begin());
  normalize_rows_inplace(q_unit, nq, dim_);

  const std::size_t nc = buckets_.size();
  std::vector<float> routing_scores(nq * nc);
  score_against_centroids(q_unit, nq, dim_, routing_centroid_matrix_, nc, routing_scores);

  const std::size_t total_n = static_cast<std::size_t>(next_id_);
  const std::size_t budget = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(params_.scan_fraction * static_cast<double>(total_n))));

  SearchResults merged;
  merged.nq = nq;
  merged.k = std::min(k, total_n);
  merged.scores.assign(nq * merged.k, 0.f);
  merged.ids.assign(nq * merged.k, 0);

  std::vector<std::vector<std::size_t>> queries_for_bucket(nc);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    std::vector<std::size_t> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      const float sa = routing_scores[qi * nc + a];
      const float sb = routing_scores[qi * nc + b];
      if (sa != sb) return sa > sb;
      return a < b;
    });

    std::size_t opened = 0;
    for (std::size_t bi : order) {
      if (buckets_[bi].count == 0) continue;
      queries_for_bucket[bi].push_back(qi);
      opened += buckets_[bi].count;
      if (opened >= budget) break;
    }
  }

  const std::size_t kk = merged.k;
  std::vector<float> heap_s(nq * kk, 0.f);
  std::vector<std::uint64_t> heap_i(nq * kk, 0);
  std::vector<std::size_t> heap_sz(nq, 0);
  std::vector<float> heap_min(nq, 0.f);
  std::vector<std::size_t> heap_mi(nq, 0);
  std::vector<float> centroid_offsets(nq, 0.f);

  for (std::size_t bi = 0; bi < nc; ++bi) {
    const auto& qis = queries_for_bucket[bi];
    if (qis.empty()) continue;
    const Bucket& b = buckets_[bi];
    if (b.count == 0) continue;
    const float* enc = encode_centroid_matrix_.data() + bi * dim_;
    for (std::size_t qi : qis) {
      centroid_offsets[qi] = dot_row(q_unit.data() + qi * dim_, enc, dim_);
    }
    score_prepared_into(prep, kk, b.blocked, b.n_blocks, b.scales, b.ids, qis, heap_s.data(),
                        heap_i.data(), heap_sz.data(), heap_min.data(), heap_mi.data(),
                        centroid_offsets.data());
  }

  heaps_to_search_results(merged, nq, kk, heap_s.data(), heap_i.data(), heap_sz.data());
  return merged;
}

}  // namespace vectorcache
