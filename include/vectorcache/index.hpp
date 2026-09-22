#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/encode/encode.hpp"
#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

class TurboQuantIndex {
 public:
  TurboQuantIndex(std::size_t dim, std::size_t bit_width);

  std::size_t dim() const { return dim_; }
  std::size_t bit_width() const { return bits_; }
  std::size_t size() const { return scales_.size(); }

  void add(std::span<const float> vectors);  // flat n*dim
  void calibrate(std::span<const float> sample);  // flat n*dim, n >= 2
  void prepare();

  SearchResults search(std::span<const float> queries, std::size_t k) const;

  std::span<const float> centroids() const { return centroids_; }
  std::span<const float> boundaries() const { return boundaries_; }
  std::span<const std::uint8_t> packed_codes() const { return packed_; }
  std::span<const float> scales() const { return scales_; }
  bool has_calibration() const { return calibration_.has_value(); }

 private:
  void ensure_blocked() const;

  std::size_t dim_;
  std::size_t bits_;
  Rotation rotation_;
  std::vector<float> boundaries_;
  std::vector<float> centroids_;
  std::optional<Calibration> calibration_;
  std::vector<std::uint8_t> packed_;
  std::vector<float> scales_;
  mutable std::vector<std::uint8_t> blocked_;
  mutable std::size_t n_blocks_ = 0;
  mutable bool blocked_ready_ = false;
  mutable std::vector<float> rotated_scratch_;
};

}  // namespace vectorcache
