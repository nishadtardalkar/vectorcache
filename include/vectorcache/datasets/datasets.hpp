#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vectorcache/datasets/reader.hpp"

namespace vectorcache::datasets {

enum class DatasetKind { Glove, OpenAi1536, OpenAi3072 };

std::optional<DatasetKind> parse_dataset_kind(const std::string& name);
std::vector<DatasetKind> all_dataset_kinds();
const char* dataset_label(DatasetKind kind);
std::size_t dataset_expected_dim(DatasetKind kind);
std::filesystem::path dataset_path(DatasetKind kind, const std::filesystem::path& data_dir);

void fetch(DatasetKind kind, const std::filesystem::path& data_dir, bool force);

class OpenDatasetReader : public DatasetReader {
 public:
  static OpenDatasetReader open(DatasetKind kind, const std::filesystem::path& data_dir,
                                DatasetSplit split);

  DatasetMeta meta() const override;
  bool next_vector_into(std::span<float> out) override;

 private:
  enum class Kind { Npy, Hdf5 };
  Kind kind_;
  std::unique_ptr<DatasetReader> reader_;
};

}  // namespace vectorcache::datasets
