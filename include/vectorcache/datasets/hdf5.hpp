#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "vectorcache/datasets/reader.hpp"

#ifdef VECTORCACHE_BUILD_GLOVE
#include <hdf5.h>
#endif

namespace vectorcache::datasets {

#ifdef VECTORCACHE_BUILD_GLOVE

/// HDF5 streaming reader for GloVe benchmark files.
class Hdf5GloveReader : public DatasetReader {
 public:
  static Hdf5GloveReader open(const std::filesystem::path& path, DatasetSplit split);
  ~Hdf5GloveReader();

  Hdf5GloveReader(const Hdf5GloveReader&) = delete;
  Hdf5GloveReader& operator=(const Hdf5GloveReader&) = delete;
  Hdf5GloveReader(Hdf5GloveReader&& other) noexcept;
  Hdf5GloveReader& operator=(Hdf5GloveReader&& other) noexcept;

  DatasetMeta meta() const override;
  bool next_vector_into(std::span<float> out) override;

 private:
  Hdf5GloveReader(hid_t file_id, hid_t dataset_id, std::string dataset_name, std::size_t dim,
                  std::size_t count);

  void ensure_chunk();
  void close();

  hid_t file_id_;
  hid_t dataset_id_;
  std::string dataset_name_;
  std::size_t dim_;
  std::size_t count_;
  std::size_t index_;
  std::vector<float> chunk_buffer_;
  std::size_t chunk_len_;
  std::size_t chunk_pos_;
};

#endif

}  // namespace vectorcache::datasets
