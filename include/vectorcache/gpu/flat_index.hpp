#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "vectorcache/search/search.hpp"

namespace vectorcache::gpu {

/// Full-index TurboQuant scan on CUDA: FWHT rotation, Lloyd–Max quantize,
/// float LUT ADC FastScan over the entire corpus (no IVF / landmarks).
class FlatIndex {
 public:
  FlatIndex(std::size_t dim, std::size_t bits);
  ~FlatIndex();

  FlatIndex(const FlatIndex&) = delete;
  FlatIndex& operator=(const FlatIndex&) = delete;
  FlatIndex(FlatIndex&&) noexcept;
  FlatIndex& operator=(FlatIndex&&) noexcept;

  std::size_t dim() const { return dim_; }
  std::size_t bits() const { return bits_; }
  std::size_t size() const { return n_vectors_; }

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
  std::size_t n_vectors_ = 0;
  std::vector<float> host_vectors_;  // staging until prepare()
  std::vector<float> boundaries_;
  std::vector<float> centroids_;
  std::string device_name_;
  mutable std::unique_ptr<DeviceState> dev_;
  bool prepared_ = false;
};

/// True if a CUDA device is available and the GPU library was built with CUDA.
bool cuda_available();

}  // namespace vectorcache::gpu
