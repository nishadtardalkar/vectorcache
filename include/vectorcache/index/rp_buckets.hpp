#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"

namespace vectorcache::index {

/// Validate coverage fraction: (0, 1].
void validate_probe_fraction(float probe_fraction);

struct BucketRange {
  std::size_t start = 0;
  std::size_t length = 0;
};

/// Spherical online clustering: grow from empty via assign + max-count splits.
class ClusterCentroids {
 public:
  ClusterCentroids() = default;
  /// Starts with zero buckets; first `assign_and_update` seeds bucket 0 from data.
  explicit ClusterCentroids(std::size_t dim);

  std::size_t num_buckets() const { return num_buckets_; }
  std::size_t dim() const { return dim_; }
  bool empty() const { return num_buckets_ == 0; }

  /// Normalized centroid row `j` (length dim).
  std::span<const float> centroid(std::size_t j) const;
  std::span<const float> centroids() const { return centroids_; }
  std::size_t count(std::size_t j) const;

  /// If empty, seed bucket 0 from `x`. Else argmax_j ⟨x, ĉ_j⟩; update S_j += x, n_j++,
  /// ĉ_j = S_j/||S_j||. Returns cell key = j.
  std::uint64_t assign_and_update(std::span<const float> x);

  /// Nearest centroid without updating (Voronoi assign).
  std::uint64_t nearest(std::span<const float> x) const;

  /// Binary-split cell `j` via diametral median cut, then local Lloyd on the two children,
  /// neighbor steal within a top-M centroid set, and global nearest eject from both children
  /// (skip moves that would empty a child). Grows `num_buckets` by 1.
  /// `vectors` is row-major N * dim; `cell_keys` length N is updated for affected members.
  void split_bucket(std::size_t j, std::span<const float> vectors,
                    std::span<std::uint64_t> cell_keys, std::size_t lloyd_iters = 1,
                    std::size_t steal_neighbors = 4);

 private:
  void normalize_centroid(std::size_t j);
  float ip_centroid(std::size_t j, std::span<const float> x) const;
  void grow_one_bucket();
  /// Argmax IP among `candidates` (non-empty). Returns candidate index into `candidates`.
  std::size_t assign_among(std::span<const float> x,
                           std::span<const std::size_t> candidates) const;
  void rebuild_sums_for_buckets(std::span<const float> vectors,
                                std::span<const std::uint64_t> cell_keys,
                                std::span<const std::size_t> buckets);
  std::vector<std::size_t> select_neighbor_buckets(std::size_t j, std::size_t j_new,
                                                   std::size_t steal_neighbors) const;

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
  /// Total indexed vectors (CSR length).
  std::size_t size() const { return offsets_.empty() ? 0 : offsets_.back(); }
  std::size_t dim() const { return dim_; }
  std::span<const float> centroids() const { return centroids_; }
  std::span<const float> centroid(std::size_t j) const;

  /// Contiguous CSR range for cell index `i` in key order (0 .. num_cells()-1).
  BucketRange cell(std::size_t i) const;

  /// Find contiguous range for an exact cell key; length 0 if missing.
  BucketRange find(std::uint64_t key) const;

  /// Walk buckets by ⟨query, ĉ_j⟩ descending until candidates cover `probe_fraction` of the
  /// index (whole buckets). Always includes at least one non-empty cell when any exist.
  /// `out_candidates` sums range lengths when non-null.
  std::vector<BucketRange> probe(std::span<const float> query, float probe_fraction,
                                 std::size_t* out_candidates = nullptr) const;

 private:
  std::vector<std::uint64_t> keys_;
  std::vector<std::size_t> offsets_;  // size keys_+1
  std::size_t num_buckets_ = 0;
  std::size_t dim_ = 0;
  AlignedVector<float> centroids_;
};

}  // namespace vectorcache::index
