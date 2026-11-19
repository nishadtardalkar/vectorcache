#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vectorcache/search/search.hpp"
#include "vectorcache/transform/landmarks.hpp"

namespace vectorcache::gpu {

/// Full-index TurboQuant scan on CUDA.
/// - `n_landmarks == 0`: FWHT + Lloyd–Max over full `dim` (no residual).
/// - `n_landmarks > 0`: ortho landmarks like CPU `LandmarkIndex` — FastScan in
///   `scan_dim = pad8(m)` with score `α·s_L + ‖q_⊥‖·‖r_i‖`.
class FlatIndex {
 public:
  FlatIndex(std::size_t dim, std::size_t bits);
  /// Ortho-landmark mode (mirrors CPU `LandmarkIndex`).
  FlatIndex(std::size_t dim, std::size_t bits, std::size_t n_landmarks,
            std::uint64_t landmark_seed = 0, bool project_then_quant = false);
  ~FlatIndex();

  FlatIndex(const FlatIndex&) = delete;
  FlatIndex& operator=(const FlatIndex&) = delete;
  FlatIndex(FlatIndex&&) noexcept;
  FlatIndex& operator=(FlatIndex&&) noexcept;

  std::size_t dim() const { return dim_; }
  std::size_t bits() const { return bits_; }
  std::size_t size() const { return n_vectors_; }
  std::size_t n_landmarks() const { return n_landmarks_; }
  std::size_t scan_dim() const { return scan_dim_; }
  bool project_then_quant() const { return project_then_quant_; }
  bool uses_landmarks() const { return n_landmarks_ > 0; }

  /// Append row-major float vectors (`n * dim`). Encoded on GPU at `prepare()`.
  void add(std::span<const float> vectors);

  /// Quantize + pack on GPU, upload codes/scales, ready for search.
  void prepare();

  /// Full-index LUT FastScan. Queries are rotated on GPU; DB stays quantized.
  SearchResults search(std::span<const float> queries, std::size_t k) const;

  /// Device name used for this index (empty if no CUDA device).
  const std::string& device_name() const { return device_name_; }

 private:
  struct DeviceState;

  std::size_t dim_ = 0;
  std::size_t bits_ = 0;
  std::size_t n_landmarks_ = 0;
  std::size_t scan_dim_ = 0;  // == dim_ when no landmarks
  std::uint64_t landmark_seed_ = 0;
  bool project_then_quant_ = false;
  std::size_t n_vectors_ = 0;
  std::vector<float> host_vectors_;  // staging until prepare()
  std::vector<float> boundaries_;    // ambient / full-dim codebook (landmark ingest)
  std::vector<float> centroids_;
  std::vector<float> lm_boundaries_;  // scan-dim codebook
  std::vector<float> lm_centroids_;
  std::vector<float> landmark_basis_;  // dim * m column-major
  QuantizedLandmarks landmark_basis_q4_;
  std::vector<float> residual_norms_;
  std::string device_name_;
  mutable std::unique_ptr<DeviceState> dev_;
  bool prepared_ = false;
};

/// True if a CUDA device is available and the GPU library was built with CUDA.
bool cuda_available();

}  // namespace vectorcache::gpu
