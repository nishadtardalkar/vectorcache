#include "vectorcache/index/rp_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::index {
namespace {

void fill_random_unit_rows(AlignedVector<float>& out, std::size_t rows, std::size_t dim,
                           std::uint64_t seed) {
  out.assign(rows * dim, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  for (std::size_t j = 0; j < rows; ++j) {
    float* row = out.data() + j * dim;
    double energy = 0.0;
    for (std::size_t d = 0; d < dim; ++d) {
      row[d] = gauss(rng);
      energy += static_cast<double>(row[d]) * static_cast<double>(row[d]);
    }
    if (energy <= 0.0) {
      row[0] = 1.0f;
      for (std::size_t d = 1; d < dim; ++d) {
        row[d] = 0.0f;
      }
    } else {
      const float inv = static_cast<float>(1.0 / std::sqrt(energy));
      for (std::size_t d = 0; d < dim; ++d) {
        row[d] *= inv;
      }
    }
  }
}

float dot(std::span<const float> a, std::span<const float> b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
}

}  // namespace

void validate_probe_radius(std::size_t probe_radius) {
  if (probe_radius == 0 || probe_radius > kMaxProbeCells) {
    throw Error("probe_radius (nprobe) must be in 1..kMaxProbeCells");
  }
}

ClusterCentroids::ClusterCentroids(std::size_t num_buckets, std::size_t dim, std::uint64_t seed)
    : num_buckets_(num_buckets), dim_(dim) {
  if (num_buckets_ == 0 || num_buckets_ > kMaxBuckets) {
    throw Error("ClusterCentroids: num_buckets must be in 1..kMaxBuckets");
  }
  if (dim_ == 0) {
    throw Error("ClusterCentroids: dim must be > 0");
  }
  fill_random_unit_rows(centroids_, num_buckets_, dim_, seed);
  sums_.assign(num_buckets_ * dim_, 0.0f);
  counts_.assign(num_buckets_, 0);
}

std::span<const float> ClusterCentroids::centroid(std::size_t j) const {
  if (j >= num_buckets_) {
    throw Error("ClusterCentroids::centroid: index out of range");
  }
  return {centroids_.data() + j * dim_, dim_};
}

std::size_t ClusterCentroids::count(std::size_t j) const {
  if (j >= num_buckets_) {
    throw Error("ClusterCentroids::count: index out of range");
  }
  return counts_[j];
}

float ClusterCentroids::ip_centroid(std::size_t j, std::span<const float> x) const {
  return dot(centroid(j), x);
}

void ClusterCentroids::normalize_centroid(std::size_t j) {
  float* c = centroids_.data() + j * dim_;
  const float* s = sums_.data() + j * dim_;
  double energy = 0.0;
  for (std::size_t d = 0; d < dim_; ++d) {
    energy += static_cast<double>(s[d]) * static_cast<double>(s[d]);
  }
  if (energy <= 0.0) {
    // Keep previous ĉ (empty / degenerate sum).
    return;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(energy));
  for (std::size_t d = 0; d < dim_; ++d) {
    c[d] = s[d] * inv;
  }
}

std::uint64_t ClusterCentroids::nearest(std::span<const float> x) const {
  if (empty()) {
    throw Error("ClusterCentroids::nearest: empty");
  }
  if (x.size() != dim_) {
    throw Error("ClusterCentroids::nearest: dim mismatch");
  }
  std::size_t best = 0;
  float best_ip = ip_centroid(0, x);
  for (std::size_t j = 1; j < num_buckets_; ++j) {
    const float ip = ip_centroid(j, x);
    if (ip > best_ip) {
      best_ip = ip;
      best = j;
    }
  }
  return static_cast<std::uint64_t>(best);
}

std::uint64_t ClusterCentroids::assign_and_update(std::span<const float> x) {
  const std::uint64_t key = nearest(x);
  const std::size_t j = static_cast<std::size_t>(key);
  float* s = sums_.data() + j * dim_;
  for (std::size_t d = 0; d < dim_; ++d) {
    s[d] += x[d];
  }
  ++counts_[j];
  normalize_centroid(j);
  return key;
}

void ClusterCentroids::rebalance(std::span<const float> vectors, std::span<std::uint64_t> cell_keys) {
  if (empty()) {
    throw Error("ClusterCentroids::rebalance: empty");
  }
  if (cell_keys.empty()) {
    return;
  }
  if (vectors.size() != cell_keys.size() * dim_) {
    throw Error("ClusterCentroids::rebalance: vectors size mismatch");
  }

  // Clear sums/counts then rebuild from nearest ĉ.
  sums_.assign(num_buckets_ * dim_, 0.0f);
  counts_.assign(num_buckets_, 0);

  for (std::size_t i = 0; i < cell_keys.size(); ++i) {
    const std::span<const float> x(vectors.data() + i * dim_, dim_);
    const std::uint64_t key = nearest(x);
    const std::size_t j = static_cast<std::size_t>(key);
    cell_keys[i] = key;
    float* s = sums_.data() + j * dim_;
    for (std::size_t d = 0; d < dim_; ++d) {
      s[d] += x[d];
    }
    ++counts_[j];
  }

  for (std::size_t j = 0; j < num_buckets_; ++j) {
    if (counts_[j] > 0) {
      normalize_centroid(j);
    }
    // Empty buckets: keep prior ĉ (already in centroids_).
  }
}

BucketIndex BucketIndex::build(std::span<const std::uint64_t> sorted_keys,
                               ClusterCentroids centroids) {
  if (sorted_keys.empty()) {
    throw Error("BucketIndex::build: empty keys");
  }
  if (centroids.empty()) {
    throw Error("BucketIndex::build: empty ClusterCentroids");
  }
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] < sorted_keys[i - 1]) {
      throw Error("BucketIndex::build: keys must be sorted ascending");
    }
  }

  BucketIndex idx;
  idx.num_buckets_ = centroids.num_buckets();
  idx.dim_ = centroids.dim();
  idx.centroids_.assign(centroids.centroids().begin(), centroids.centroids().end());
  idx.keys_.reserve(sorted_keys.size());
  idx.offsets_.reserve(sorted_keys.size() + 1);
  idx.offsets_.push_back(0);

  std::uint64_t cur = sorted_keys[0];
  idx.keys_.push_back(cur);
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] != cur) {
      idx.offsets_.push_back(i);
      cur = sorted_keys[i];
      idx.keys_.push_back(cur);
    }
  }
  idx.offsets_.push_back(sorted_keys.size());
  return idx;
}

std::span<const float> BucketIndex::centroid(std::size_t j) const {
  if (j >= num_buckets_) {
    throw Error("BucketIndex::centroid: index out of range");
  }
  return {centroids_.data() + j * dim_, dim_};
}

BucketRange BucketIndex::cell(std::size_t i) const {
  if (i >= keys_.size()) {
    throw Error("BucketIndex::cell: index out of range");
  }
  const std::size_t start = offsets_[i];
  const std::size_t end = offsets_[i + 1];
  return {start, end - start};
}

BucketRange BucketIndex::find(std::uint64_t key) const {
  const auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
  if (it == keys_.end() || *it != key) {
    return {};
  }
  const std::size_t cell = static_cast<std::size_t>(it - keys_.begin());
  const std::size_t start = offsets_[cell];
  const std::size_t end = offsets_[cell + 1];
  return {start, end - start};
}

std::vector<BucketRange> BucketIndex::probe(std::span<const float> query, std::size_t probe_radius,
                                            std::size_t* out_candidates) const {
  if (empty()) {
    throw Error("BucketIndex::probe: empty index");
  }
  if (query.size() != dim_) {
    throw Error("BucketIndex::probe: query dim mismatch");
  }
  validate_probe_radius(probe_radius);
  const std::size_t nprobe = std::min(probe_radius, num_buckets_);

  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(num_buckets_);
  for (std::size_t j = 0; j < num_buckets_; ++j) {
    const float ip = dot(centroid(j), query);
    scored.emplace_back(ip, static_cast<std::uint64_t>(j));
  }
  const std::size_t sort_n = nprobe;
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(sort_n),
                    scored.end(),
                    [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<BucketRange> ranges;
  ranges.reserve(nprobe);
  std::size_t candidates = 0;
  for (std::size_t i = 0; i < nprobe; ++i) {
    const BucketRange range = find(scored[i].second);
    if (range.length == 0) {
      continue;
    }
    ranges.push_back(range);
    candidates += range.length;
  }
  if (out_candidates != nullptr) {
    *out_candidates = candidates;
  }
  return ranges;
}

}  // namespace vectorcache::index
