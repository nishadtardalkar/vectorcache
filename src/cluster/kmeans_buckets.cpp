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

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

float dot_row(const float* a, const float* b, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    s += static_cast<double>(a[d]) * static_cast<double>(b[d]);
  }
  return static_cast<float>(s);
}

void normalize_inplace(float* row, std::size_t dim) {
  double s = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    s += static_cast<double>(row[d]) * static_cast<double>(row[d]);
  }
  const float inv = (s > static_cast<double>(kMinInputNorm) * kMinInputNorm)
                        ? static_cast<float>(1.0 / std::sqrt(s))
                        : 0.f;
  for (std::size_t d = 0; d < dim; ++d) row[d] *= inv;
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

double mean_sq_dist_to_centroid(std::span<const float> floats, std::size_t n, std::size_t dim,
                                const float* centroid) {
  if (n == 0) return 0.0;
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float cos = dot_row(floats.data() + i * dim, centroid, dim);
    sum += 2.0 * (1.0 - static_cast<double>(cos));
  }
  return sum;
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
  if (!(params_.var_threshold > 0.f)) {
    throw std::invalid_argument("var_threshold must be > 0");
  }
  if (params_.min_split_size < 2) {
    throw std::invalid_argument("min_split_size must be >= 2");
  }
  if (params_.split_iters < 1) {
    throw std::invalid_argument("split_iters must be >= 1");
  }
  if (params_.max_bucket_size != 0 && params_.max_bucket_size < params_.min_split_size) {
    throw std::invalid_argument("max_bucket_size must be 0 (disabled) or >= min_split_size");
  }
  auto cb = codebook(bits_, dim_);
  boundaries_ = std::move(cb.first);
  codebook_centroids_ = std::move(cb.second);

  Bucket seed;
  seed.centroid.assign(dim_, 0.f);
  buckets_.push_back(std::move(seed));
  sync_centroid_matrix();
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

float BucketedTurboQuantIndex::bucket_variance(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_variance");
  return buckets_[bucket].variance();
}

std::size_t BucketedTurboQuantIndex::bucket_size(std::size_t bucket) const {
  if (bucket >= buckets_.size()) throw std::out_of_range("bucket_size");
  return buckets_[bucket].count;
}

void BucketedTurboQuantIndex::update_centroid_online(Bucket& b, std::span<const float> unit_row) {
  const float cos_old = (b.count == 0) ? 1.f : dot_row(unit_row.data(), b.centroid.data(), dim_);
  const double n = static_cast<double>(b.count);
  if (b.count == 0) {
    std::copy(unit_row.begin(), unit_row.end(), b.centroid.begin());
    b.count = 1;
    b.sum_sq_dist = 0.0;
    return;
  }
  for (std::size_t d = 0; d < dim_; ++d) {
    b.centroid[d] = static_cast<float>((b.centroid[d] * n + unit_row[d]) / (n + 1.0));
  }
  normalize_inplace(b.centroid.data(), dim_);
  ++b.count;
  // Track Σ 2(1 − cos) vs centroid at assignment time (stable streaming estimate).
  b.sum_sq_dist += 2.0 * (1.0 - static_cast<double>(cos_old));
}

void BucketedTurboQuantIndex::reencode_bucket(Bucket& b) {
  b.packed.clear();
  b.scales.clear();
  if (b.count == 0) {
    b.blocked.clear();
    b.n_blocks = 0;
    b.blocked_ready = true;
    return;
  }
  const Calibration* cal = calibration_ ? &*calibration_ : nullptr;
  encode(b.floats, b.count, dim_, rotation_, boundaries_, codebook_centroids_, bits_, cal,
         rotated_scratch_, b.packed, b.scales);
  b.blocked_ready = false;
}

void BucketedTurboQuantIndex::append_to_bucket(std::size_t bi, std::span<const float> unit_row,
                                              std::uint64_t id) {
  Bucket& b = buckets_[bi];
  const std::size_t old_n = b.count;
  b.floats.resize((old_n + 1) * dim_);
  std::copy(unit_row.begin(), unit_row.end(), b.floats.begin() + static_cast<std::ptrdiff_t>(old_n * dim_));
  b.ids.push_back(id);

  // Encode single row into this bucket's packed/scales.
  const Calibration* cal = calibration_ ? &*calibration_ : nullptr;
  encode(unit_row, 1, dim_, rotation_, boundaries_, codebook_centroids_, bits_, cal,
         rotated_scratch_, b.packed, b.scales);
  b.blocked_ready = false;

  update_centroid_online(b, unit_row);
}

void BucketedTurboQuantIndex::split_bucket(std::size_t bi) {
  Bucket& parent = buckets_[bi];
  const std::size_t n = parent.count;
  if (n < 2) return;

  // Take ownership of parent storage so children aren't layered on top of it.
  std::vector<float> parent_floats = std::move(parent.floats);
  std::vector<std::uint64_t> parent_ids = std::move(parent.ids);
  parent.packed.clear();
  parent.packed.shrink_to_fit();
  parent.scales.clear();
  parent.scales.shrink_to_fit();
  parent.blocked.clear();
  parent.blocked.shrink_to_fit();
  parent.blocked_ready = false;
  parent.n_blocks = 0;
  const std::vector<float> parent_centroid = parent.centroid;

  // Seed: farthest from centroid, then farthest from first seed.
  std::size_t s0 = 0;
  float worst0 = std::numeric_limits<float>::infinity();
  for (std::size_t i = 0; i < n; ++i) {
    const float cos = dot_row(parent_floats.data() + i * dim_, parent_centroid.data(), dim_);
    if (cos < worst0) {
      worst0 = cos;
      s0 = i;
    }
  }
  std::size_t s1 = (s0 + 1) % n;
  float worst1 = std::numeric_limits<float>::infinity();
  const float* seed0 = parent_floats.data() + s0 * dim_;
  for (std::size_t i = 0; i < n; ++i) {
    if (i == s0) continue;
    const float cos = dot_row(parent_floats.data() + i * dim_, seed0, dim_);
    if (cos < worst1) {
      worst1 = cos;
      s1 = i;
    }
  }

  std::vector<float> c0(dim_), c1(dim_);
  std::copy(parent_floats.begin() + static_cast<std::ptrdiff_t>(s0 * dim_),
            parent_floats.begin() + static_cast<std::ptrdiff_t>((s0 + 1) * dim_), c0.begin());
  std::copy(parent_floats.begin() + static_cast<std::ptrdiff_t>(s1 * dim_),
            parent_floats.begin() + static_cast<std::ptrdiff_t>((s1 + 1) * dim_), c1.begin());

  std::vector<std::uint8_t> assign(n, 0);
  for (std::size_t iter = 0; iter < params_.split_iters; ++iter) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < static_cast<int>(n); ++i) {
      const float* row = parent_floats.data() + static_cast<std::size_t>(i) * dim_;
      const float d0 = dot_row(row, c0.data(), dim_);
      const float d1 = dot_row(row, c1.data(), dim_);
      assign[static_cast<std::size_t>(i)] = (d1 > d0) ? 1 : 0;
    }
    std::vector<double> sum0(dim_, 0.0), sum1(dim_, 0.0);
    std::size_t n0 = 0, n1 = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float* row = parent_floats.data() + i * dim_;
      if (assign[i] == 0) {
        ++n0;
        for (std::size_t d = 0; d < dim_; ++d) sum0[d] += row[d];
      } else {
        ++n1;
        for (std::size_t d = 0; d < dim_; ++d) sum1[d] += row[d];
      }
    }
    if (n0 == 0 || n1 == 0) {
      // Degenerate: force a balanced cut by index.
      for (std::size_t i = 0; i < n; ++i) assign[i] = (i < n / 2) ? 0 : 1;
      std::fill(sum0.begin(), sum0.end(), 0.0);
      std::fill(sum1.begin(), sum1.end(), 0.0);
      n0 = n1 = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const float* row = parent_floats.data() + i * dim_;
        if (assign[i] == 0) {
          ++n0;
          for (std::size_t d = 0; d < dim_; ++d) sum0[d] += row[d];
        } else {
          ++n1;
          for (std::size_t d = 0; d < dim_; ++d) sum1[d] += row[d];
        }
      }
    }
    for (std::size_t d = 0; d < dim_; ++d) {
      c0[d] = static_cast<float>(sum0[d] / static_cast<double>(n0));
      c1[d] = static_cast<float>(sum1[d] / static_cast<double>(n1));
    }
    normalize_inplace(c0.data(), dim_);
    normalize_inplace(c1.data(), dim_);
  }

  Bucket child0, child1;
  child0.centroid = std::move(c0);
  child1.centroid = std::move(c1);
  child0.floats.reserve((n / 2 + 1) * dim_);
  child1.floats.reserve((n / 2 + 1) * dim_);
  child0.ids.reserve(n / 2 + 1);
  child1.ids.reserve(n / 2 + 1);

  for (std::size_t i = 0; i < n; ++i) {
    Bucket& dest = (assign[i] == 0) ? child0 : child1;
    dest.floats.insert(dest.floats.end(),
                       parent_floats.begin() + static_cast<std::ptrdiff_t>(i * dim_),
                       parent_floats.begin() + static_cast<std::ptrdiff_t>((i + 1) * dim_));
    dest.ids.push_back(parent_ids[i]);
    ++dest.count;
  }
  {
    std::vector<float>().swap(parent_floats);
    std::vector<std::uint64_t>().swap(parent_ids);
  }
  child0.sum_sq_dist =
      mean_sq_dist_to_centroid(child0.floats, child0.count, dim_, child0.centroid.data());
  child1.sum_sq_dist =
      mean_sq_dist_to_centroid(child1.floats, child1.count, dim_, child1.centroid.data());

  reencode_bucket(child0);
  reencode_bucket(child1);

  buckets_[bi] = std::move(child0);
  buckets_.push_back(std::move(child1));
  sync_centroid_matrix();
}

void BucketedTurboQuantIndex::maybe_split(std::size_t bi) {
  while (bi < buckets_.size() && buckets_[bi].count >= params_.min_split_size) {
    const Bucket& b = buckets_[bi];
    const bool var_split = b.variance() >= params_.var_threshold;
    const bool size_split =
        params_.max_bucket_size > 0 && b.count > params_.max_bucket_size;
    if (!var_split && !size_split) break;
    const std::size_t n_before = buckets_.size();
    split_bucket(bi);
    if (buckets_.size() > n_before) {
      maybe_split(buckets_.size() - 1);
    }
  }
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

  // Normalize one row at a time — avoid a second full n×dim float matrix.
  std::vector<float> row(dim_);

  // Stream one vector at a time so variance splits see an up-to-date centroid matrix.
  for (std::size_t i = 0; i < n; ++i) {
    copy_normalize_row(vectors.subspan(i * dim_, dim_), row.data(), dim_);

    // First vector ever: seed the single empty bucket centroid.
    if (next_id_ == 0 && buckets_.size() == 1 && buckets_[0].count == 0) {
      std::copy(row.begin(), row.end(), buckets_[0].centroid.begin());
      sync_centroid_matrix();
    }

    const std::size_t nc = buckets_.size();
    std::uint32_t best_j = 0;
    float best = -std::numeric_limits<float>::infinity();
    if (next_id_ == 0) {
      best_j = 0;
    } else {
      for (std::size_t j = 0; j < nc; ++j) {
        if (buckets_[j].count == 0) continue;
        const float s = dot_row(row.data(), centroid_matrix_.data() + j * dim_, dim_);
        if (s > best) {
          best = s;
          best_j = static_cast<std::uint32_t>(j);
        }
      }
    }

    const std::uint64_t id = next_id_++;
    append_to_bucket(best_j, std::span<const float>(row.data(), dim_), id);
    std::copy(buckets_[best_j].centroid.begin(), buckets_[best_j].centroid.end(),
              centroid_matrix_.begin() + static_cast<std::ptrdiff_t>(best_j * dim_));
    maybe_split(best_j);
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

  // Unit-normalize queries for centroid scoring (input space).
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

  std::vector<SearchResults> locals(nq);
  for (std::size_t bi = 0; bi < nc; ++bi) {
    const auto& qis = queries_for_bucket[bi];
    if (qis.empty()) continue;
    const Bucket& b = buckets_[bi];
    SearchResults partial =
        score_prepared(prep, merged.k, b.blocked, b.n_blocks, b.scales, b.ids, qis);
    if (partial.k == 0) continue;

    for (std::size_t i = 0; i < qis.size(); ++i) {
      const std::size_t qi = qis[i];
      SearchResults one;
      one.nq = 1;
      one.k = partial.k;
      one.scores.assign(partial.scores.begin() + static_cast<std::ptrdiff_t>(i * partial.k),
                        partial.scores.begin() + static_cast<std::ptrdiff_t>((i + 1) * partial.k));
      one.ids.assign(partial.ids.begin() + static_cast<std::ptrdiff_t>(i * partial.k),
                     partial.ids.begin() + static_cast<std::ptrdiff_t>((i + 1) * partial.k));
      if (locals[qi].nq == 0) {
        locals[qi] = std::move(one);
      } else {
        if (locals[qi].k < merged.k) {
          locals[qi].scores.resize(merged.k, 0.f);
          locals[qi].ids.resize(merged.k, 0);
          locals[qi].k = merged.k;
        }
        merge_search_results(locals[qi], one);
      }
    }
  }

  for (std::size_t qi = 0; qi < nq; ++qi) {
    const std::size_t kk = std::min(merged.k, locals[qi].k);
    for (std::size_t j = 0; j < kk; ++j) {
      merged.scores[qi * merged.k + j] = locals[qi].scores[j];
      merged.ids[qi * merged.k + j] = locals[qi].ids[j];
    }
  }

  return merged;
}

}  // namespace vectorcache
