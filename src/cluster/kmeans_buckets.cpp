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
  if (!(params_.cos_threshold > -1.f && params_.cos_threshold <= 1.f)) {
    throw std::invalid_argument("cos_threshold must be in (-1, 1]");
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

void BucketedTurboQuantIndex::sync_centroid_matrix() {
  centroid_matrix_.resize(buckets_.size() * dim_);
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    std::copy(buckets_[i].centroid.begin(), buckets_[i].centroid.end(),
              centroid_matrix_.begin() + static_cast<std::ptrdiff_t>(i * dim_));
  }
}

std::size_t BucketedTurboQuantIndex::bucket_size(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_size");
  return buckets_[bucket].count;
}

std::span<const float> BucketedTurboQuantIndex::bucket_centroid(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_centroid");
  return buckets_[bucket].centroid;
}

std::size_t BucketedTurboQuantIndex::spawn_bucket(std::span<const float> unit_row) {
  Bucket b;
  b.centroid.assign(unit_row.begin(), unit_row.end());
  buckets_.push_back(std::move(b));
  sync_centroid_matrix();
  return buckets_.size() - 1;
}

void BucketedTurboQuantIndex::append_to_bucket(std::size_t bi, std::span<const float> unit_row,
                                              std::uint64_t id) {
  Bucket& b = buckets_[bi];
  const std::size_t old_n = b.count;

  // IVF residual: encode r = x̂ − c (TurboQuant re-normalizes r → r̂ + α ≈ ‖r‖).
  std::vector<float> residual(dim_);
  for (std::size_t d = 0; d < dim_; ++d) {
    residual[d] = unit_row[d] - b.centroid[d];
  }

  b.floats.resize((old_n + 1) * dim_);
  std::copy(residual.begin(), residual.end(),
            b.floats.begin() + static_cast<std::ptrdiff_t>(old_n * dim_));
  b.ids.push_back(id);
  ++b.count;

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

    std::size_t best_j = 0;
    float best = -std::numeric_limits<float>::infinity();
    const std::size_t nc = buckets_.size();
    for (std::size_t j = 0; j < nc; ++j) {
      const float s = dot_row(row.data(), centroid_matrix_.data() + j * dim_, dim_);
      if (s > best) {
        best = s;
        best_j = j;
      }
    }

    if (nc == 0 || best < params_.cos_threshold) {
      best_j = spawn_bucket(std::span<const float>(row.data(), dim_));
    }

    const std::uint64_t id = next_id_++;
    append_to_bucket(best_j, std::span<const float>(row.data(), dim_), id);
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
  for (std::size_t i = 0; i < buckets_.size(); ++i) {
    ensure_bucket_blocked(i);
    // Search only needs blocked + scales + ids; drop ingest scratch.
    Bucket& b = buckets_[i];
    b.floats.clear();
    b.floats.shrink_to_fit();
    b.packed.clear();
    b.packed.shrink_to_fit();
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

  // Unit-normalize queries for centroid scoring (input space). Same q̂ feeds FastScan LUTs
  // once; per opened bucket we only add ⟨q̂, c⟩ (IVF residual estimator).
  std::vector<float> q_unit(nq * dim_);
  std::copy(queries.begin(), queries.end(), q_unit.begin());
  normalize_rows_inplace(q_unit, nq, dim_);

  const std::size_t nc = buckets_.size();
  std::vector<float> bucket_scores(nq * nc);
  score_against_centroids(q_unit, nq, dim_, centroid_matrix_, nc, bucket_scores);

  const std::size_t total_n = static_cast<std::size_t>(next_id_);
  const std::size_t budget = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::ceil(params_.scan_fraction * static_cast<double>(total_n))));

  SearchResults merged;
  merged.nq = nq;
  merged.k = std::min(k, total_n);
  merged.scores.assign(nq * merged.k, 0.f);
  merged.ids.assign(nq * merged.k, 0);

  // Per-query open lists (respect scan_fraction), then invert so each bucket is FastScan'd
  // once for the batch of queries that need it — keeps multi-query SIMD/OpenMP.
  std::vector<std::vector<std::size_t>> queries_for_bucket(nc);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    std::vector<std::size_t> order(nc);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      const float sa = bucket_scores[qi * nc + a];
      const float sb = bucket_scores[qi * nc + b];
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
  // Per-query ⟨q̂, c⟩ for the bucket currently being scored (reused buffer).
  std::vector<float> centroid_offsets(nq, 0.f);

  for (std::size_t bi = 0; bi < nc; ++bi) {
    const auto& qis = queries_for_bucket[bi];
    if (qis.empty()) continue;
    const Bucket& b = buckets_[bi];
    if (b.count == 0) continue;
    for (std::size_t qi : qis) {
      centroid_offsets[qi] = bucket_scores[qi * nc + bi];
    }
    score_prepared_into(prep, kk, b.blocked, b.n_blocks, b.scales, b.ids, qis, heap_s.data(),
                        heap_i.data(), heap_sz.data(), heap_min.data(), heap_mi.data(),
                        centroid_offsets.data());
  }

  heaps_to_search_results(merged, nq, kk, heap_s.data(), heap_i.data(), heap_sz.data());
  return merged;
}

}  // namespace vectorcache
