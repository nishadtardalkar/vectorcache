#include "vectorcache/index/rp_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::index {
namespace {

constexpr int kSplitIters = 5;

float dot(std::span<const float> a, std::span<const float> b) {
  double sum = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    sum += static_cast<double>(a[i]) * static_cast<double>(b[i]);
  }
  return static_cast<float>(sum);
}

void normalize_row(float* row, std::size_t dim) {
  double energy = 0.0;
  for (std::size_t d = 0; d < dim; ++d) {
    energy += static_cast<double>(row[d]) * static_cast<double>(row[d]);
  }
  if (energy <= 0.0) {
    row[0] = 1.0f;
    for (std::size_t d = 1; d < dim; ++d) {
      row[d] = 0.0f;
    }
    return;
  }
  const float inv = static_cast<float>(1.0 / std::sqrt(energy));
  for (std::size_t d = 0; d < dim; ++d) {
    row[d] *= inv;
  }
}

}  // namespace

void validate_probe_fraction(float probe_fraction) {
  if (!(probe_fraction > 0.0f) || probe_fraction > 1.0f || !std::isfinite(probe_fraction)) {
    throw Error("probe_fraction must be in (0, 1]");
  }
}

ClusterCentroids::ClusterCentroids(std::size_t dim) : dim_(dim) {
  if (dim_ == 0) {
    throw Error("ClusterCentroids: dim must be > 0");
  }
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

void ClusterCentroids::grow_one_bucket() {
  const std::size_t old = num_buckets_;
  ++num_buckets_;
  centroids_.resize(num_buckets_ * dim_, 0.0f);
  sums_.resize(num_buckets_ * dim_, 0.0f);
  counts_.resize(num_buckets_, 0);
  if (old == 0) {
    return;
  }
  // New row already zero-filled; caller fills sums/centroid.
  (void)old;
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
  if (dim_ == 0) {
    throw Error("ClusterCentroids::assign_and_update: uninitialized");
  }
  if (x.size() != dim_) {
    throw Error("ClusterCentroids::assign_and_update: dim mismatch");
  }

  if (empty()) {
    grow_one_bucket();
    float* s = sums_.data();
    float* c = centroids_.data();
    for (std::size_t d = 0; d < dim_; ++d) {
      s[d] = x[d];
      c[d] = x[d];
    }
    normalize_row(c, dim_);
    counts_[0] = 1;
    return 0;
  }

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

void ClusterCentroids::split_bucket(std::size_t j, std::span<const float> vectors,
                                    std::span<std::uint64_t> cell_keys) {
  if (j >= num_buckets_) {
    throw Error("ClusterCentroids::split_bucket: index out of range");
  }
  if (cell_keys.empty()) {
    return;
  }
  if (vectors.size() != cell_keys.size() * dim_) {
    throw Error("ClusterCentroids::split_bucket: vectors size mismatch");
  }

  std::vector<std::size_t> members;
  members.reserve(counts_[j]);
  for (std::size_t i = 0; i < cell_keys.size(); ++i) {
    if (cell_keys[i] == static_cast<std::uint64_t>(j)) {
      members.push_back(i);
    }
  }
  if (members.size() < 2) {
    return;
  }

  // Init: c0 = current ĉ, c1 = farthest member from ĉ (min IP).
  std::vector<float> c0(centroids_.data() + j * dim_, centroids_.data() + (j + 1) * dim_);
  std::vector<float> c1(dim_, 0.0f);
  {
    float worst_ip = 2.0f;
    std::size_t farthest = members[0];
    for (std::size_t mi : members) {
      const std::span<const float> x(vectors.data() + mi * dim_, dim_);
      const float ip = dot(c0, x);
      if (ip < worst_ip) {
        worst_ip = ip;
        farthest = mi;
      }
    }
    const float* src = vectors.data() + farthest * dim_;
    std::copy(src, src + dim_, c1.begin());
    normalize_row(c1.data(), dim_);
  }

  std::vector<std::uint8_t> side(members.size(), 0);
  for (int iter = 0; iter < kSplitIters; ++iter) {
    for (std::size_t m = 0; m < members.size(); ++m) {
      const std::span<const float> x(vectors.data() + members[m] * dim_, dim_);
      const float ip0 = dot(c0, x);
      const float ip1 = dot(c1, x);
      side[m] = (ip1 > ip0) ? 1 : 0;
    }

    std::fill(c0.begin(), c0.end(), 0.0f);
    std::fill(c1.begin(), c1.end(), 0.0f);
    std::size_t n0 = 0;
    std::size_t n1 = 0;
    for (std::size_t m = 0; m < members.size(); ++m) {
      const float* x = vectors.data() + members[m] * dim_;
      float* dst = (side[m] == 0) ? c0.data() : c1.data();
      for (std::size_t d = 0; d < dim_; ++d) {
        dst[d] += x[d];
      }
      if (side[m] == 0) {
        ++n0;
      } else {
        ++n1;
      }
    }
    if (n0 == 0 || n1 == 0) {
      // Degenerate: force a 50/50 cut by index order.
      std::fill(c0.begin(), c0.end(), 0.0f);
      std::fill(c1.begin(), c1.end(), 0.0f);
      n0 = 0;
      n1 = 0;
      for (std::size_t m = 0; m < members.size(); ++m) {
        side[m] = (m * 2 < members.size()) ? 0 : 1;
        const float* x = vectors.data() + members[m] * dim_;
        float* dst = (side[m] == 0) ? c0.data() : c1.data();
        for (std::size_t d = 0; d < dim_; ++d) {
          dst[d] += x[d];
        }
        if (side[m] == 0) {
          ++n0;
        } else {
          ++n1;
        }
      }
    }
    normalize_row(c0.data(), dim_);
    normalize_row(c1.data(), dim_);
    if (n0 == 0 || n1 == 0) {
      return;
    }
  }

  const std::size_t j_new = num_buckets_;
  grow_one_bucket();

  // Rebuild sums/counts for j and j_new from final assignment.
  float* s0 = sums_.data() + j * dim_;
  float* s1 = sums_.data() + j_new * dim_;
  std::fill(s0, s0 + dim_, 0.0f);
  std::fill(s1, s1 + dim_, 0.0f);
  counts_[j] = 0;
  counts_[j_new] = 0;

  for (std::size_t m = 0; m < members.size(); ++m) {
    const std::size_t i = members[m];
    const float* x = vectors.data() + i * dim_;
    if (side[m] == 0) {
      cell_keys[i] = static_cast<std::uint64_t>(j);
      for (std::size_t d = 0; d < dim_; ++d) {
        s0[d] += x[d];
      }
      ++counts_[j];
    } else {
      cell_keys[i] = static_cast<std::uint64_t>(j_new);
      for (std::size_t d = 0; d < dim_; ++d) {
        s1[d] += x[d];
      }
      ++counts_[j_new];
    }
  }

  normalize_centroid(j);
  normalize_centroid(j_new);
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

std::vector<BucketRange> BucketIndex::probe(std::span<const float> query, float probe_fraction,
                                            std::size_t* out_candidates) const {
  if (empty()) {
    throw Error("BucketIndex::probe: empty index");
  }
  if (query.size() != dim_) {
    throw Error("BucketIndex::probe: query dim mismatch");
  }
  validate_probe_fraction(probe_fraction);

  const std::size_t n = size();
  const std::size_t target = std::max(
      std::size_t{1},
      static_cast<std::size_t>(std::ceil(static_cast<double>(probe_fraction) * static_cast<double>(n))));

  std::vector<std::pair<float, std::uint64_t>> scored;
  scored.reserve(num_buckets_);
  for (std::size_t j = 0; j < num_buckets_; ++j) {
    const float ip = dot(centroid(j), query);
    scored.emplace_back(ip, static_cast<std::uint64_t>(j));
  }
  std::sort(scored.begin(), scored.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  std::vector<BucketRange> ranges;
  ranges.reserve(std::min(num_buckets_, target));
  std::size_t candidates = 0;
  for (const auto& [ip, key] : scored) {
    (void)ip;
    const BucketRange range = find(key);
    if (range.length == 0) {
      continue;
    }
    ranges.push_back(range);
    candidates += range.length;
    if (candidates >= target) {
      break;
    }
  }
  if (out_candidates != nullptr) {
    *out_candidates = candidates;
  }
  return ranges;
}

}  // namespace vectorcache::index
