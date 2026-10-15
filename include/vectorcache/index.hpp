#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

/// FWHT (1 round) → 4-bit scalar quant → orthonormal landmarks → shortlist → exact re-rank.
class LandmarkIndex {
 public:
  LandmarkIndex(std::size_t dim, std::size_t bit_width, std::size_t n_landmarks = kDefaultLandmarks,
                std::size_t shortlist = kDefaultShortlist, std::uint64_t landmark_seed = 0);

  std::size_t dim() const { return dim_; }
  std::size_t bit_width() const { return bits_; }
  std::size_t n_landmarks() const { return n_landmarks_; }
  std::size_t shortlist() const { return shortlist_; }
  std::size_t size() const { return n_vectors_; }

  void set_shortlist(std::size_t shortlist) { shortlist_ = shortlist; }

  /// Append flat n*dim vectors (L2-normalized or not; stored normalized).
  void add(std::span<const float> vectors);
  void prepare();

  SearchResults search(std::span<const float> queries, std::size_t k) const;

 private:
  void encode_row(const float* src, float* unit_out, float* landmark_row_out);

  std::size_t dim_;
  std::size_t bits_;
  std::size_t n_landmarks_;
  std::size_t shortlist_;
  std::uint64_t landmark_seed_;
  Rotation rotation_;
  std::vector<float> boundaries_;
  std::vector<float> centroids_;
  std::vector<float> landmark_basis_;  // dim * m, column-major
  std::vector<float> db_unit_;         // n * dim (L2-normalized originals for re-rank)
  std::vector<float> landmark_rows_;   // n * m (row-major before prepare)
  mutable std::vector<float> blocked_landmarks_;  // n_blocks * m * kBlock
  mutable std::size_t n_blocks_ = 0;
  std::size_t n_vectors_ = 0;
  mutable bool prepared_ = false;
  mutable std::vector<float> rotate_scratch_;
};

/// Back-compat name used by older call sites / benches.
using TurboQuantIndex = LandmarkIndex;

}  // namespace vectorcache
