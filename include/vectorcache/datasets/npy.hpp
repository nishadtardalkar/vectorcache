#pragma once

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "vectorcache/datasets/reader.hpp"

namespace vectorcache::datasets {

namespace detail {

struct MmapRegion {
  void* data = nullptr;
  std::size_t size = 0;

  MmapRegion() = default;
  MmapRegion(MmapRegion&& other) noexcept;
  MmapRegion& operator=(MmapRegion&& other) noexcept;
  ~MmapRegion();

  MmapRegion(const MmapRegion&) = delete;
  MmapRegion& operator=(const MmapRegion&) = delete;
};

MmapRegion mmap_file(const std::filesystem::path& path);

}  // namespace detail

/// Memory-mapped streaming reader for NumPy .npy float32 matrices shaped (N, dim).
class NpyReader : public DatasetReader {
 public:
  static NpyReader open(const std::filesystem::path& path, const char* label);

  DatasetMeta meta() const override;
  bool next_vector_into(std::span<float> out) override;

  /// Zero-copy view of the next vector when backed by mmap or owned storage.
  std::optional<std::span<const float>> next_vector_span();

 private:
  NpyReader(std::vector<char> owned_data, detail::MmapRegion mmap, const char* label,
            std::size_t dim, std::size_t count, std::size_t data_offset);

  const char* data_ptr() const;
  std::size_t data_size() const;

  std::vector<char> owned_data_;
  detail::MmapRegion mmap_;
  const char* label_;
  std::size_t dim_;
  std::size_t count_;
  std::size_t index_;
  std::size_t data_offset_;
};

/// Write a float32 NPY matrix (rows x cols) to path.
void write_npy_f32_matrix(const std::filesystem::path& path, std::size_t rows, std::size_t cols,
                          const float* data);

}  // namespace vectorcache::datasets
