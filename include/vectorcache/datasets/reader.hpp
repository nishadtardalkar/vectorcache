#pragma once

#include <cstddef>
#include <span>
#include <string>

namespace vectorcache::datasets {

enum class DatasetSplit { Train, Test };

inline const char* hdf5_dataset_name(DatasetSplit split) {
  return split == DatasetSplit::Train ? "train" : "test";
}

struct DatasetMeta {
  std::size_t dim = 0;
  std::size_t count = 0;
  const char* label = "";
};

class DatasetReader {
 public:
  virtual ~DatasetReader() = default;
  virtual DatasetMeta meta() const = 0;
  virtual bool next_vector_into(std::span<float> out) = 0;
};

}  // namespace vectorcache::datasets
