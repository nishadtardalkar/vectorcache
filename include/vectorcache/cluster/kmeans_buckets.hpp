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
  float var_threshold = 0.5f;
  std::size_t min_split_size = 256;
  std::size_t split_iters = 5;
};

/// Streaming cosine k-means IVF in front of per-bucket TurboQuant FastScan.
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
  float bucket_variance(std::size_t bucket) const;
  std::size_t bucket_size(std::size_t bucket) const;

 private:
  struct Bucket {
    std::vector<float> floats;  // unit rows, n * dim
    std::vector<std::uint8_t> packed;
    std::vector<float> scales;
    std::vector<std::uint64_t> ids;
    mutable std::vector<std::uint8_t> blocked;
    mutable std::size_t n_blocks = 0;
    mutable bool blocked_ready = false;

    std::vector<float> centroid;  // unit, dim
    std::size_t count = 0;
    double sum_sq_dist = 0.0;  // Σ ‖x − c‖² with unit vectors = Σ 2(1 − cos)

    float variance() const {
      return count == 0 ? 0.f : static_cast<float>(sum_sq_dist / static_cast<double>(count));
    }
  };

  void ensure_bucket_blocked(std::size_t bi) const;
  void sync_centroid_matrix();
  void append_to_bucket(std::size_t bi, std::span<const float> unit_row, std::uint64_t id);
  void update_centroid_online(Bucket& b, std::span<const float> unit_row);
  void maybe_split(std::size_t bi);
  void split_bucket(std::size_t bi);
  void reencode_bucket(Bucket& b);

  std::size_t dim_;
  std::size_t bits_;
  BucketParams params_;
  Rotation rotation_;
  std::vector<float> boundaries_;
  std::vector<float> codebook_centroids_;
  std::optional<Calibration> calibration_;
  mutable std::vector<float> rotated_scratch_;

  std::vector<Bucket> buckets_;
  std::vector<float> centroid_matrix_;  // n_buckets * dim
  std::uint64_t next_id_ = 0;
  bool prepared_ = false;
};

}  // namespace vectorcache
