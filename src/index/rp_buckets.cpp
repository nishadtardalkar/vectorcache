#include "vectorcache/index/rp_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::index {
namespace {

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

std::size_t ClusterCentroids::assign_among(std::span<const float> x,
                                           std::span<const std::size_t> candidates) const {
  if (candidates.empty()) {
    throw Error("ClusterCentroids::assign_among: empty candidates");
  }
  std::size_t best = 0;
  float best_ip = ip_centroid(candidates[0], x);
  for (std::size_t c = 1; c < candidates.size(); ++c) {
    const float ip = ip_centroid(candidates[c], x);
    if (ip > best_ip) {
      best_ip = ip;
      best = c;
    }
  }
  return best;
}

void ClusterCentroids::rebuild_sums_for_buckets(std::span<const float> vectors,
                                                std::span<const std::uint64_t> cell_keys,
                                                std::span<const std::size_t> buckets) {
  for (std::size_t b : buckets) {
    float* s = sums_.data() + b * dim_;
    std::fill(s, s + dim_, 0.0f);
    counts_[b] = 0;
  }
  // Membership scan once; accumulate only for requested buckets.
  for (std::size_t i = 0; i < cell_keys.size(); ++i) {
    const std::size_t key = static_cast<std::size_t>(cell_keys[i]);
    bool wanted = false;
    for (std::size_t b : buckets) {
      if (b == key) {
        wanted = true;
        break;
      }
    }
    if (!wanted) {
      continue;
    }
    const float* x = vectors.data() + i * dim_;
    float* s = sums_.data() + key * dim_;
    for (std::size_t d = 0; d < dim_; ++d) {
      s[d] += x[d];
    }
    ++counts_[key];
  }
  for (std::size_t b : buckets) {
    normalize_centroid(b);
  }
}

std::vector<std::size_t> ClusterCentroids::select_neighbor_buckets(
    std::size_t j, std::size_t j_new, std::size_t steal_neighbors) const {
  if (steal_neighbors == 0 || num_buckets_ <= 2) {
    return {};
  }
  struct Scored {
    float score;
    std::size_t idx;
  };
  std::vector<Scored> scored;
  scored.reserve(num_buckets_ >= 2 ? num_buckets_ - 2 : 0);
  for (std::size_t n = 0; n < num_buckets_; ++n) {
    if (n == j || n == j_new) {
      continue;
    }
    const float ip_j = dot(centroid(n), centroid(j));
    const float ip_new = dot(centroid(n), centroid(j_new));
    scored.push_back({std::max(ip_j, ip_new), n});
  }
  const std::size_t take = std::min(steal_neighbors, scored.size());
  if (take == 0) {
    return {};
  }
  std::partial_sort(scored.begin(), scored.begin() + static_cast<std::ptrdiff_t>(take),
                    scored.end(),
                    [](const Scored& a, const Scored& b) {
                      if (a.score != b.score) {
                        return a.score > b.score;
                      }
                      return a.idx < b.idx;
                    });
  std::vector<std::size_t> out;
  out.reserve(take);
  for (std::size_t i = 0; i < take; ++i) {
    out.push_back(scored[i].idx);
  }
  return out;
}

void ClusterCentroids::split_bucket(std::size_t j, std::span<const float> vectors,
                                    std::span<std::uint64_t> cell_keys, std::size_t lloyd_iters,
                                    std::size_t steal_neighbors) {
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

  // Diametral axis: A = farthest from ĉ, B = farthest from A.
  const std::span<const float> c(centroids_.data() + j * dim_, dim_);
  std::size_t i_a = members[0];
  {
    float worst_ip = 2.0f;
    for (std::size_t mi : members) {
      const std::span<const float> x(vectors.data() + mi * dim_, dim_);
      const float ip = dot(c, x);
      if (ip < worst_ip) {
        worst_ip = ip;
        i_a = mi;
      }
    }
  }
  const std::span<const float> a(vectors.data() + i_a * dim_, dim_);
  std::size_t i_b = members[0] == i_a ? members[1] : members[0];
  {
    float worst_ip = 2.0f;
    for (std::size_t mi : members) {
      if (mi == i_a) {
        continue;
      }
      const std::span<const float> x(vectors.data() + mi * dim_, dim_);
      const float ip = dot(a, x);
      if (ip < worst_ip) {
        worst_ip = ip;
        i_b = mi;
      }
    }
  }
  const std::span<const float> b(vectors.data() + i_b * dim_, dim_);

  // Score s = ⟨x, B⟩ − ⟨x, A⟩; median-cut for ~50/50 sides.
  struct Scored {
    float s;
    std::size_t idx;  // global vector index
  };
  std::vector<Scored> scored;
  scored.reserve(members.size());
  for (std::size_t mi : members) {
    const std::span<const float> x(vectors.data() + mi * dim_, dim_);
    scored.push_back({dot(x, b) - dot(x, a), mi});
  }
  std::sort(scored.begin(), scored.end(), [](const Scored& u, const Scored& v) {
    if (u.s != v.s) {
      return u.s < v.s;
    }
    return u.idx < v.idx;
  });

  const std::size_t n = scored.size();
  const std::size_t n0 = n / 2;  // lower ⌊N/2⌋ → side 0; upper ⌈N/2⌉ → side 1

  const std::size_t j_new = num_buckets_;
  grow_one_bucket();

  float* s0 = sums_.data() + j * dim_;
  float* s1 = sums_.data() + j_new * dim_;
  std::fill(s0, s0 + dim_, 0.0f);
  std::fill(s1, s1 + dim_, 0.0f);
  counts_[j] = 0;
  counts_[j_new] = 0;

  for (std::size_t m = 0; m < n; ++m) {
    const std::size_t i = scored[m].idx;
    const float* x = vectors.data() + i * dim_;
    if (m < n0) {
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

  if (counts_[j] < 1 || counts_[j_new] < 1) {
    return;
  }

  const std::size_t children_arr[2] = {j, j_new};
  const std::span<const std::size_t> children(children_arr, 2);

  // Local Lloyd: reassign children members among {j, j_new}.
  for (std::size_t iter = 0; iter < lloyd_iters; ++iter) {
    for (std::size_t i = 0; i < cell_keys.size(); ++i) {
      const std::size_t key = static_cast<std::size_t>(cell_keys[i]);
      if (key != j && key != j_new) {
        continue;
      }
      const std::span<const float> x(vectors.data() + i * dim_, dim_);
      const std::size_t best = assign_among(x, children);
      cell_keys[i] = static_cast<std::uint64_t>(children[best]);
    }
    rebuild_sums_for_buckets(vectors, cell_keys, children);
    if (counts_[j] < 1 || counts_[j_new] < 1) {
      return;
    }
  }

  // Neighbor steal: reassign members of S among S only.
  auto neighbors = select_neighbor_buckets(j, j_new, steal_neighbors);
  if (neighbors.empty()) {
    return;
  }

  std::vector<std::size_t> S;
  S.reserve(2 + neighbors.size());
  S.push_back(j);
  S.push_back(j_new);
  S.insert(S.end(), neighbors.begin(), neighbors.end());

  auto in_S = [&](std::size_t key) {
    for (std::size_t b : S) {
      if (b == key) {
        return true;
      }
    }
    return false;
  };

  for (std::size_t i = 0; i < cell_keys.size(); ++i) {
    const std::size_t old_key = static_cast<std::size_t>(cell_keys[i]);
    if (!in_S(old_key)) {
      continue;
    }
    const std::span<const float> x(vectors.data() + i * dim_, dim_);
    const std::size_t best = assign_among(x, S);
    const std::size_t new_key = S[best];
    if (new_key == old_key) {
      continue;
    }
    const float* xv = vectors.data() + i * dim_;
    float* s_old = sums_.data() + old_key * dim_;
    float* s_new = sums_.data() + new_key * dim_;
    for (std::size_t d = 0; d < dim_; ++d) {
      s_old[d] -= xv[d];
      s_new[d] += xv[d];
    }
    if (counts_[old_key] > 0) {
      --counts_[old_key];
    }
    ++counts_[new_key];
    cell_keys[i] = static_cast<std::uint64_t>(new_key);
  }

  for (std::size_t b : S) {
    normalize_centroid(b);
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
