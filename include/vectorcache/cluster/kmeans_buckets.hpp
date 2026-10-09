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

/// Online dense soft400 / 2-means partition / absorb / pulse densify (product fat champ).
/// Defaults match experiments soft309_trig1016 (r@1≈0.981 @ ~3105×322 on SIFT1M @ 2% scan).
struct BucketParams {
  float scan_fraction = 0.1f;
  /// Hard fission when bucket count reaches this (champ fission_cap).
  std::size_t max_bucket_size = 800;
  /// Soft energy-split threshold outside the densify pulse.
  std::size_t energy_soft = 400;
  float energy_trig = 0.095f;
  /// Margin-weighted routing update + seam attract.
  float margin_tau = 0.025f;
  float attract_margin = 0.035f;
  float attract = 0.012f;
  /// Tiny-list absorb into near rival.
  std::size_t absorb_max = 48;
  float absorb_ip = 0.90f;
  /// End-of-ingest asymmetric anti-rival push on routing centroids only.
  float rival_push = 0.08f;
  /// Stream schedule as fractions of expected_n (SIFT1M champ: 0.65 / 0.78 / 0.90).
  /// When expected_n==0, pulse densify / recovery / late absorb are disabled.
  std::size_t expected_n = 0;
  float pulse_start_frac = 0.65f;
  float pulse_end_frac = 0.78f;
  float late_absorb_frac = 0.90f;
  std::size_t pulse_soft = 309;
  float pulse_trig = 0.1016f;
  std::size_t ramp_soft0 = 400;
  std::size_t ramp_soft1 = 415;
  std::size_t fat_soft_n = 280;
  std::size_t fat_soft_extra = 20;
  std::size_t recov_absorb_max = 96;
  std::size_t recov_well_min = 200;
  float recov_absorb_ip = 0.93f;
  std::size_t late_absorb_max = 60;
  std::size_t late_well_min = 250;
  float late_absorb_ip = 0.96f;

  /// Deprecated no-ops kept for API compatibility with older callers.
  float cos_var_threshold = 0.02f;
  std::size_t min_bucket_size = 256;
};

/// Online nearest-centroid IVF in front of per-bucket TurboQuant FastScan.
/// Routing centroids use margin-weighted means + soft seam attract; fission is
/// 1-iter spherical 2-means Voronoi partition (hard @ max_bucket_size, soft on
/// energy schedule with optional mid-stream densify pulse). Tiny near-duplicate
/// lists absorb into rivals; prepare() applies asymmetric anti-rival push to
/// routing centroids only. Encode centroids stay frozen at bucket birth /
/// partition for residual codes `x̂ − encode_c`.
/// Search opens buckets by routing score until `scan_fraction` of N, then ranks
/// with `⟨q̂, encode_c⟩ + α · ⟨q̂, r̂⟩`.
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
  /// Finalize FastScan layouts, bake rival-asymmetric routing, drop ingest scratch.
  void prepare();

  SearchResults search(std::span<const float> queries, std::size_t k) const;

  bool has_calibration() const { return calibration_.has_value(); }
  std::size_t bucket_size(std::size_t bucket) const;
  /// Unit routing centroid. Length = dim().
  std::span<const float> bucket_centroid(std::size_t bucket) const;
  /// Unit encode centroid (frozen at birth / partition). Length = dim().
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
    std::vector<float> routing_centroid;  // unit, dim — margin-weighted mean
    std::vector<double> sum;              // weighted vector sum, dim
    double wsum = 0.0;
    std::size_t count = 0;
    double cos_mean = 0.0;
    double cos_m2 = 0.0;
  };

  void ensure_bucket_blocked(std::size_t bi) const;
  void sync_centroid_matrices();
  void sync_routing_row(std::size_t bi);
  void sync_encode_row(std::size_t bi);
  std::size_t spawn_bucket(std::span<const float> unit_row);
  void append_member(std::size_t bi, std::span<const float> unit_row, std::uint64_t id,
                     float margin);
  void soft_pull(std::size_t bi, std::span<const float> unit_row);
  int nearest_rival(std::size_t bi) const;
  void kill_bucket(std::size_t bi);
  bool maybe_absorb(std::size_t bi);
  void absorb_into(std::size_t keep, std::size_t drop);
  void split_2means(std::size_t bi);
  void finalize_partition(Bucket& b, std::vector<float> units, std::vector<std::uint64_t> ids);
  void encode_units_into(Bucket& b);
  void bake_rival_asym();
  std::size_t soft_threshold(std::size_t stream_i, std::size_t count) const;
  float soft_trig(std::size_t stream_i) const;

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
  std::size_t stream_i_ = 0;  // vectors ingested so far (champ stream index)
  bool prepared_ = false;
};

}  // namespace vectorcache
