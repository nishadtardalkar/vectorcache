#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"

namespace vectorcache::index {

inline constexpr std::size_t kMaxProbeCells = 4096;
inline constexpr std::size_t kMaxBuckets = 65536;

/// Validate nprobe count: 1..kMaxProbeCells.
void validate_probe_radius(std::size_t probe_radius);

struct BucketRange {
  std::size_t start = 0;
  std::size_t length = 0;
};

/// Spherical online clustering: random unit centroids, assign by max IP, exact running mean.
class ClusterCentroids {
 public:
  ClusterCentroids() = default;
  ClusterCentroids(std::size_t num_buckets, std::size_t dim, std::uint64_t seed);

  std::size_t num_buckets() const { return num_buckets_; }
  std::size_t dim() const { return dim_; }
  bool empty() const { return num_buckets_ == 0; }

  /// Normalized centroid row `j` (length dim).
  std::span<const float> centroid(std::size_t j) const;
  std::span<const float> centroids() const { return centroids_; }
  std::size_t count(std::size_t j) const;

  /// Argmax_j ⟨x, ĉ_j⟩; update S_j += x, n_j++, ĉ_j = S_j/||S_j||. Returns cell key = j.
  std::uint64_t assign_and_update(std::span<const float> x);

  /// Nearest centroid without updating (Voronoi assign).
  std::uint64_t nearest(std::span<const float> x) const;

  /// Keep current ĉ as assign targets; reassign all rows; recompute S/n/ĉ from members.
  /// `vectors` is row-major N * dim; `cell_keys` length N is overwritten.
  void rebalance(std::span<const float> vectors, std::span<std::uint64_t> cell_keys);

 private:
  void normalize_centroid(std::size_t j);
  float ip_centroid(std::size_t j, std::span<const float> x) const;

  std::size_t num_buckets_ = 0;
  std::size_t dim_ = 0;
  /// B * dim normalized centroids.
  AlignedVector<float> centroids_;
  /// B * dim unnormalized sums.
  AlignedVector<float> sums_;
  std::vector<std::size_t> counts_;
};

/// CSR over cell keys (bucket ids). Store rows were permuted by key.
class BucketIndex {
 public:
  BucketIndex() = default;

  /// `sorted_keys[i]` is the cell key of store row i after argsort-by-key permute.
  /// Takes final normalized centroids from `centroids` (copies ĉ matrix).
  static BucketIndex build(std::span<const std::uint64_t> sorted_keys, ClusterCentroids centroids);

  bool empty() const { return keys_.empty(); }
  std::size_t num_cells() const { return keys_.size(); }
  std::size_t num_buckets() const { return num_buckets_; }
  std::size_t dim() const { return dim_; }
  std::span<const float> centroids() const { return centroids_; }
  std::span<const float> centroid(std::size_t j) const;

  /// Contiguous CSR range for cell index `i` in key order (0 .. num_cells()-1).
  BucketRange cell(std::size_t i) const;

  /// Find contiguous range for an exact cell key; length 0 if missing.
  BucketRange find(std::uint64_t key) const;

  /// Top-`probe_radius` buckets by ⟨query, ĉ_j⟩ (descending). Only non-empty CSR cells.
  /// `out_candidates` sums range lengths when non-null.
  std::vector<BucketRange> probe(std::span<const float> query, std::size_t probe_radius,
                                 std::size_t* out_candidates = nullptr) const;

 private:
  std::vector<std::uint64_t> keys_;
  std::vector<std::size_t> offsets_;  // size keys_+1
  std::size_t num_buckets_ = 0;
  std::size_t dim_ = 0;
  AlignedVector<float> centroids_;
};

}  // namespace vectorcache::index
