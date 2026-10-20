#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

/// FWHT → quant → orthonormal landmarks → FastScan top-k with residual-energy bound.
/// Single scan (no full-dim re-rank). Score = s_L + ‖q_⊥‖·‖r_i‖.
/// `scan_bits` = landmark FastScan; `rerank_bits` kept for ABI (unused at search).
class LandmarkIndex {
 public:
  LandmarkIndex(std::size_t dim, std::size_t scan_bits, std::size_t n_landmarks = kDefaultLandmarks,
                std::uint64_t landmark_seed = 0, std::size_t rerank_bits = 0);

  std::size_t dim() const { return dim_; }
  /// Landmark FastScan bit width (alias of scan_bits).
  std::size_t bit_width() const { return scan_bits_; }
  std::size_t scan_bits() const { return scan_bits_; }
  std::size_t rerank_bits() const { return rerank_bits_; }
  std::size_t n_landmarks() const { return n_landmarks_; }
  std::size_t scan_dim() const { return scan_dim_; }
  std::size_t size() const { return n_vectors_; }

  /// Append flat n*dim vectors (L2-normalized or not).
  void add(std::span<const float> vectors);
  void prepare();

  SearchResults search(std::span<const float> queries, std::size_t k) const;

 private:
  void encode_row(const float* src, float* unit_out, float* landmark_row_out, float* residual_out);

  std::size_t dim_;
  std::size_t scan_bits_;    // landmark FastScan quant
  std::size_t rerank_bits_;  // unused at search (ABI / ingest codebook for ŷ)
  std::size_t n_landmarks_;
  std::size_t scan_dim_;  // n_landmarks padded to multiple of 8 for FastScan
  std::uint64_t landmark_seed_;
  Rotation rotation_;
  std::vector<float> boundaries_;       // full-dim FWHT quant (for ŷ → landmarks)
  std::vector<float> centroids_;
  std::vector<float> lm_boundaries_;    // landmark-space FastScan quant (scan_bits)
  std::vector<float> lm_centroids_;
  std::vector<float> landmark_basis_;  // dim * m, column-major
  std::vector<float> db_unit_;         // n * dim (scratch until prepare)
  std::vector<float> landmark_rows_;   // n * scan_dim (row-major before prepare)
  std::vector<float> residual_norms_;  // ‖r‖ of (α·ŷ) orthogonal to landmarks
  std::vector<std::uint8_t> blocked_codes_;  // landmark FastScan
  std::vector<float> scales_;                // landmark α
  /// Interleaved [α_i, ‖r_i‖] pairs for FastScan flush (size 2·n).
  std::vector<float> scale_resid_;
  mutable std::size_t n_blocks_ = 0;
  std::size_t n_vectors_ = 0;
  mutable bool prepared_ = false;
  mutable std::vector<float> rotate_scratch_;
};

using TurboQuantIndex = LandmarkIndex;

}  // namespace vectorcache
