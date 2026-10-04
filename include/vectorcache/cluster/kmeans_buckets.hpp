#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/encode/encode.hpp"
#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

struct BucketParams {
  float scan_fraction = 0.1f;
  /// Split when sample variance of assignment cosines reaches this (and size >= min).
  float cos_var_threshold = 0.02f;
  /// Minimum members before a variance-triggered split is allowed.
  std::size_t min_bucket_size = 1024;
  /// Hard cap: always split when count reaches this.
  std::size_t max_bucket_size = 2048;
};

/// Online nearest-centroid IVF in front of per-bucket TurboQuant FastScan.
/// Routing centroids track the exact normalized running mean; encode centroids
/// stay frozen at bucket birth for residual codes `x̂ − encode_c`. Buckets split
/// when cosine-score variance exceeds `cos_var_threshold` (with size ≥ min) or
/// when size hits `max_bucket_size`.
/// Search opens buckets by routing centroid score until `scan_fraction` of N,
/// then ranks with `⟨q̂, encode_c⟩ + α · ⟨q̂, r̂⟩`.
class BucketedTurboQuantIndex {
 public:
  BucketedTurboQuantIndex(std::size_t dim, std::size_t bit_width, BucketParams params = {});

  std::size_t dim() const { return dim_; }
  std::size_t bit_width() const { return bits_; }
  std::size_t size() const { return next_id_; }
  std::size_t num_buckets() const { return buckets_.size(); }
  const BucketParams& params() const { return params_; }

  void calibrate(std::span<const float> sample);
  void add(std::span<const float> vectors);
  /// Finalize FastScan layouts and drop ingest-only float / packed scratch.
  void prepare();

  SearchResults search(std::span<const float> queries, std::size_t k) const;

  bool has_calibration() const { return calibration_.has_value(); }
  std::size_t bucket_size(std::size_t bucket) const;
  /// Unit routing centroid (normalized running mean). Length = dim().
  std::span<const float> bucket_centroid(std::size_t bucket) const;
  /// Unit encode centroid (frozen at birth). Length = dim().
  std::span<const float> bucket_encode_centroid(std::size_t bucket) const;
  /// Sample variance of assignment cosines for bucket (`0` if count < 2).
  double bucket_cos_variance(std::size_t bucket) const;

 private:
  struct Bucket {
    std::vector<float> floats;  // unit rows (x̂), n * dim — ingest scratch for split
    std::vector<std::uint8_t> packed;
    std::vector<float> scales;
    std::vector<std::uint64_t> ids;
    mutable std::vector<std::uint8_t> blocked;
    mutable std::size_t n_blocks = 0;
    mutable bool blocked_ready = false;

    std::vector<float> encode_centroid;   // unit, dim — frozen at create / split
    std::vector<float> routing_centroid;  // unit, dim — normalized running mean
    std::vector<double> sum;              // running vector sum, dim
    std::size_t count = 0;
    double cos_mean = 0.0;
    double cos_m2 = 0.0;
  };

  void ensure_bucket_blocked(std::size_t bi) const;
  void sync_centroid_matrices();
  void sync_routing_row(std::size_t bi);
  void sync_encode_row(std::size_t bi);
  std::size_t spawn_bucket(std::span<const float> unit_row);
  void append_to_bucket(std::size_t bi, std::span<const float> unit_row, std::uint64_t id);
  bool needs_split(const Bucket& b) const;
  void split_bucket(std::size_t bi);
  void finalize_partition(Bucket& b, std::vector<float> units, std::vector<std::uint64_t> ids);
  void encode_units_into(Bucket& b);

  std::size_t dim_;
  std::size_t bits_;
  BucketParams params_;
  Rotation rotation_;
  std::vector<float> boundaries_;
  std::vector<float> codebook_centroids_;
  std::optional<Calibration> calibration_;
  mutable std::vector<float> rotated_scratch_;

  std::vector<Bucket> buckets_;
  std::vector<float> routing_centroid_matrix_;  // n_buckets * dim
  std::vector<float> encode_centroid_matrix_;   // n_buckets * dim
  std::uint64_t next_id_ = 0;
  bool prepared_ = false;
};

}  // namespace vectorcache
