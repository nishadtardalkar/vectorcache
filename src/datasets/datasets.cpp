#include "vectorcache/datasets/datasets.hpp"

#include "vectorcache/datasets/hdf5.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/error.hpp"

namespace vectorcache::datasets {

namespace {

constexpr const char* kGloveFilename = "glove-200-angular.hdf5";
constexpr const char* kSift1MFilename = "sift-128-euclidean.hdf5";

}  // namespace

std::optional<DatasetKind> parse_dataset_kind(const std::string& name) {
  if (name == "glove") return DatasetKind::Glove;
  if (name == "sift1m" || name == "sift") return DatasetKind::Sift1M;
  if (name == "openai-1536") return DatasetKind::OpenAi1536;
  if (name == "openai-3072") return DatasetKind::OpenAi3072;
  return std::nullopt;
}

std::vector<DatasetKind> all_dataset_kinds() {
  return {DatasetKind::Glove, DatasetKind::Sift1M, DatasetKind::OpenAi1536,
          DatasetKind::OpenAi3072};
}

const char* dataset_label(DatasetKind kind) {
  switch (kind) {
    case DatasetKind::Glove:
      return "glove";
    case DatasetKind::Sift1M:
      return "sift1m";
    case DatasetKind::OpenAi1536:
      return "openai-1536";
    case DatasetKind::OpenAi3072:
      return "openai-3072";
  }
  return "";
}

std::filesystem::path dataset_path(DatasetKind kind, const std::filesystem::path& data_dir) {
  switch (kind) {
    case DatasetKind::Glove:
      return data_dir / kGloveFilename;
    case DatasetKind::Sift1M:
      return data_dir / kSift1MFilename;
    case DatasetKind::OpenAi1536:
      return data_dir / "openai-1536.npy";
    case DatasetKind::OpenAi3072:
      return data_dir / "openai-3072.npy";
  }
  return data_dir;
}

OpenDatasetReader OpenDatasetReader::open(DatasetKind kind, const std::filesystem::path& data_dir,
                                        DatasetSplit split) {
  OpenDatasetReader reader;
  const auto path = dataset_path(kind, data_dir);

  switch (kind) {
    case DatasetKind::Glove:
    case DatasetKind::Sift1M:
#ifdef VECTORCACHE_BUILD_GLOVE
      if (!std::filesystem::is_regular_file(path)) {
        throw Error(std::string(dataset_label(kind)) + " dataset not found: " + path.string());
      }
      reader.kind_ = Kind::Hdf5;
      reader.reader_ = std::make_unique<Hdf5GloveReader>(
          Hdf5GloveReader::open(path, split, dataset_label(kind)));
      return reader;
#else
      throw Error(
          "HDF5 dataset ingestion requires VECTORCACHE_BUILD_GLOVE (HDF5 C library must be "
          "installed)");
#endif
    case DatasetKind::OpenAi1536:
    case DatasetKind::OpenAi3072:
      if (split == DatasetSplit::Test) {
        throw Error("OpenAI NPY datasets do not have a test split");
      }
      if (!std::filesystem::is_regular_file(path)) {
        throw Error("OpenAI dataset not found: " + path.string());
      }
      reader.kind_ = Kind::Npy;
      reader.reader_ =
          std::make_unique<NpyReader>(NpyReader::open(path, dataset_label(kind)));
      return reader;
  }

  return reader;
}

DatasetMeta OpenDatasetReader::meta() const { return reader_->meta(); }

bool OpenDatasetReader::next_vector_into(std::span<float> out) {
  return reader_->next_vector_into(out);
}

}  // namespace vectorcache::datasets
