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
  if (!(params_.cos_var_threshold > 0.f)) {
    throw std::invalid_argument("cos_var_threshold must be > 0");
  }
  if (params_.min_bucket_size < 2) {
    throw std::invalid_argument("min_bucket_size must be >= 2");
  }
  if (params_.max_bucket_size < params_.min_bucket_size) {
    throw std::invalid_argument("max_bucket_size must be >= min_bucket_size");
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

bool BucketedTurboQuantIndex::needs_split(const Bucket& b) const {
  if (b.count >= params_.max_bucket_size) return true;
  if (b.count < params_.min_bucket_size || b.count < 2) return false;
  const double var = b.cos_m2 / static_cast<double>(b.count - 1);
  return var >= static_cast<double>(params_.cos_var_threshold);
}

std::size_t BucketedTurboQuantIndex::spawn_bucket(std::span<const float> unit_row) {
  Bucket b;
  // Freeze encode centroid at founding vector; append fills sum/count/routing.
  b.encode_centroid.assign(unit_row.begin(), unit_row.end());
  b.routing_centroid.assign(unit_row.begin(), unit_row.end());
  b.sum.assign(dim_, 0.0);
  b.count = 0;
  b.cos_mean = 0.0;
  b.cos_m2 = 0.0;
  buckets_.push_back(std::move(b));
  sync_centroid_matrices();
  return buckets_.size() - 1;
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
  for (std::size_t i = 0; i < n; ++i) {
    const float* row = b.floats.data() + i * dim_;
    for (std::size_t d = 0; d < dim_; ++d) b.sum[d] += static_cast<double>(row[d]);
  }
  b.encode_centroid.resize(dim_);
  b.routing_centroid.resize(dim_);
  if (n == 0) {
    std::fill(b.encode_centroid.begin(), b.encode_centroid.end(), 0.f);
    std::fill(b.routing_centroid.begin(), b.routing_centroid.end(), 0.f);
    b.cos_mean = 0.0;
    b.cos_m2 = 0.0;
    encode_units_into(b);
    return;
  }
  normalize_from_sum(b.sum.data(), b.encode_centroid.data(), dim_);
  b.routing_centroid = b.encode_centroid;

  // Recompute Welford cosine variance vs routing centroid.
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

void BucketedTurboQuantIndex::split_bucket(std::size_t bi) {
  // Bound recursive re-splits (max size halves each successful bipartition).
  constexpr std::size_t kMaxSplitDepth = 32;
  std::size_t depth = 0;
  std::vector<std::size_t> stack;
  stack.push_back(bi);

  while (!stack.empty() && depth < kMaxSplitDepth) {
    const std::size_t cur = stack.back();
    stack.pop_back();
    Bucket& parent = buckets_[cur];
    if (!needs_split(parent) || parent.count < 2) continue;

    const std::size_t n = parent.count;
    const float* seed_a = parent.routing_centroid.data();

    // Seed B = member with lowest cosine to routing mean.
    std::size_t far_i = 0;
    float far_cos = std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < n; ++i) {
      const float cos = dot_row(parent.floats.data() + i * dim_, seed_a, dim_);
      if (cos < far_cos) {
        far_cos = cos;
        far_i = i;
      }
    }
    const float* seed_b = parent.floats.data() + far_i * dim_;

    std::vector<std::uint8_t> side(n, 0);
    std::size_t n_a = 0;
    std::size_t n_b = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const float* row = parent.floats.data() + i * dim_;
      const float ca = dot_row(row, seed_a, dim_);
      const float cb = dot_row(row, seed_b, dim_);
      if (cb > ca) {
        side[i] = 1;
        ++n_b;
      } else {
        side[i] = 0;
        ++n_a;
      }
    }

    if (n_a == 0 || n_b == 0) {
      // Median split by cosine to seed A.
      std::vector<std::pair<float, std::size_t>> order(n);
      for (std::size_t i = 0; i < n; ++i) {
        order[i] = {dot_row(parent.floats.data() + i * dim_, seed_a, dim_), i};
      }
      std::nth_element(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(n / 2),
                       order.end(),
                       [](const auto& x, const auto& y) { return x.first < y.first; });
      std::fill(side.begin(), side.end(), static_cast<std::uint8_t>(0));
      n_a = 0;
      n_b = 0;
      for (std::size_t r = 0; r < n / 2; ++r) {
        side[order[r].second] = 1;
        ++n_b;
      }
      n_a = n - n_b;
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
    sync_routing_row(cur);
    sync_encode_row(cur);

    Bucket child;
    finalize_partition(child, std::move(units_b), std::move(ids_b));
    buckets_.push_back(std::move(child));
    const std::size_t child_i = buckets_.size() - 1;
    // Matrices grow; resync all rows (cheap relative to re-encode).
    sync_centroid_matrices();

    ++depth;
    if (needs_split(buckets_[cur])) stack.push_back(cur);
    if (needs_split(buckets_[child_i])) stack.push_back(child_i);
  }
}

void BucketedTurboQuantIndex::append_to_bucket(std::size_t bi, std::span<const float> unit_row,
                                              std::uint64_t id) {
  Bucket& b = buckets_[bi];

  // Cosine vs current routing centroid (before mean update), then Welford.
  const float cos = (b.count == 0) ? 1.f : dot_row(unit_row.data(), b.routing_centroid.data(), dim_);
  {
    const double c = static_cast<double>(cos);
    const double n_new = static_cast<double>(b.count + 1);
    const double delta = c - b.cos_mean;
    b.cos_mean += delta / n_new;
    b.cos_m2 += delta * (c - b.cos_mean);
  }

  // Exact running mean → normalized routing centroid.
  if (b.sum.size() != dim_) b.sum.assign(dim_, 0.0);
  for (std::size_t d = 0; d < dim_; ++d) b.sum[d] += static_cast<double>(unit_row[d]);
  ++b.count;
  b.routing_centroid.resize(dim_);
  normalize_from_sum(b.sum.data(), b.routing_centroid.data(), dim_);
  sync_routing_row(bi);

  // Store unit row; encode residual vs frozen encode centroid.
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

    std::size_t best_j = 0;
    if (buckets_.empty()) {
      best_j = spawn_bucket(std::span<const float>(row.data(), dim_));
    } else {
      float best_cos = -std::numeric_limits<float>::infinity();
      const std::size_t nc = buckets_.size();
      for (std::size_t j = 0; j < nc; ++j) {
        if (buckets_[j].count == 0) continue;
        const float cos = dot_row(row.data(), routing_centroid_matrix_.data() + j * dim_, dim_);
        if (cos > best_cos) {
          best_cos = cos;
          best_j = j;
        }
      }
    }

    const std::uint64_t id = next_id_++;
    append_to_bucket(best_j, std::span<const float>(row.data(), dim_), id);
    if (needs_split(buckets_[best_j])) {
      split_bucket(best_j);
    }
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
    // Search only needs blocked + scales + ids + centroids; drop ingest scratch.
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

  // Unit-normalize queries for centroid scoring (input space).
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

  // Per-query open lists (respect scan_fraction), then invert so each bucket is FastScan'd
  // once for the batch of queries that need it — keeps multi-query SIMD/OpenMP.
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
  // Per-query ⟨q̂, encode_c⟩ for the bucket currently being scored.
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
