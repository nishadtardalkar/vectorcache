#include "vectorcache/datasets/datasets.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>

#include <curl/curl.h>

#ifdef VECTORCACHE_FETCH_OPENAI
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#endif

#include "vectorcache/datasets/hdf5.hpp"
#include "vectorcache/datasets/npy.hpp"
#include "vectorcache/error.hpp"

namespace vectorcache::datasets {

namespace {

constexpr const char* kGloveUrl = "http://ann-benchmarks.com/glove-200-angular.hdf5";
constexpr const char* kGloveFilename = "glove-200-angular.hdf5";
constexpr std::uint64_t kGloveMinBytes = 100'000'000;
constexpr const char* kHfDatasetBase = "https://huggingface.co/datasets";

std::size_t openai_parquet_shard_count(std::size_t dim) {
  switch (dim) {
    case 1536:
      return 26;
    case 3072:
      return 63;
    default:
      throw Error("unsupported OpenAI embedding dimension: " + std::to_string(dim));
  }
}

size_t curl_write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
  auto* out = static_cast<std::vector<char>*>(userdata);
  const size_t total = size * nmemb;
  out->insert(out->end(), ptr, ptr + total);
  return total;
}

void download_url_to_file(const std::string& url, const std::filesystem::path& tmp,
                          const std::filesystem::path& dest) {
  if (tmp.has_parent_path()) {
    std::filesystem::create_directories(tmp.parent_path());
  }

  CURL* curl = curl_easy_init();
  if (!curl) {
    throw Error("failed to initialize libcurl");
  }

  std::vector<char> buffer;
  curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 3600L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_callback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

  const CURLcode res = curl_easy_perform(curl);
  long http_code = 0;
  curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
  curl_easy_cleanup(curl);

  if (res != CURLE_OK || http_code >= 400) {
    throw Error("failed to download " + url);
  }

  std::ofstream out(tmp, std::ios::binary);
  if (!out) {
    throw Error("failed to create temporary file " + tmp.string());
  }
  out.write(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  out.close();

  std::filesystem::rename(tmp, dest);
}

void validate_glove(const std::filesystem::path& path) {
  if (!std::filesystem::is_regular_file(path)) {
    throw Error("file does not exist: " + path.string());
  }
  if (std::filesystem::file_size(path) < kGloveMinBytes) {
    throw Error("GloVe file is too small");
  }

  std::ifstream file(path, std::ios::binary);
  char header[8] = {};
  file.read(header, 8);
  static constexpr char kHdf5Sig[] = "\x89HDF\r\n\x1a\n";
  if (std::memcmp(header, kHdf5Sig, 8) != 0) {
    throw Error("file does not look like HDF5: " + path.string());
  }

#ifdef VECTORCACHE_BUILD_GLOVE
  Hdf5GloveReader::open(path, DatasetSplit::Train);
  Hdf5GloveReader::open(path, DatasetSplit::Test);
#endif
}

std::pair<std::size_t, std::size_t> read_openai_shape(const std::filesystem::path& path) {
  auto reader = NpyReader::open(path, "openai");
  return {reader.meta().count, reader.meta().dim};
}

void validate_openai(const std::filesystem::path& path, std::size_t dim) {
  const auto [rows, cols] = read_openai_shape(path);
  if (rows != 1'000'000 || cols != dim) {
    throw Error("unexpected OpenAI shape");
  }
}

void print_dataset_status(const std::string& label, const std::filesystem::path& path,
                          const std::string& detail, bool skipped) {
  const double size_mb =
      static_cast<double>(std::filesystem::file_size(path)) / (1024.0 * 1024.0);
  const char* prefix = skipped ? "[skip]" : "Saved";
  std::printf("%s %s: %s (%s, %.0f MB)\n", prefix, label.c_str(), path.string().c_str(),
              detail.c_str(), size_mb);
}

#ifdef VECTORCACHE_FETCH_OPENAI
void append_embedding_values_as_f32(const std::shared_ptr<arrow::Array>& values,
                                    std::vector<float>& out) {
  if (auto floats = std::dynamic_pointer_cast<arrow::FloatArray>(values)) {
    const float* raw = floats->raw_values();
    out.insert(out.end(), raw, raw + floats->length());
    return;
  }
  if (auto doubles = std::dynamic_pointer_cast<arrow::DoubleArray>(values)) {
    const double* raw = doubles->raw_values();
    for (int64_t i = 0; i < doubles->length(); ++i) {
      out.push_back(static_cast<float>(raw[i]));
    }
    return;
  }
  throw Error("expected float or double values inside embedding list");
}

void append_embeddings_from_chunk(const std::shared_ptr<arrow::Array>& chunk, std::size_t dim,
                                  const std::filesystem::path& path, std::vector<float>& out) {
  if (auto list_array = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunk)) {
    if (static_cast<std::size_t>(list_array->value_length()) != dim) {
      throw Error("unexpected embedding width in " + path.string());
    }
    append_embedding_values_as_f32(list_array->values(), out);
    return;
  }

  if (auto list_array = std::dynamic_pointer_cast<arrow::ListArray>(chunk)) {
    if (list_array->length() > 0 &&
        static_cast<std::size_t>(list_array->value_length(0)) != dim) {
      throw Error("unexpected embedding width in " + path.string());
    }
    append_embedding_values_as_f32(list_array->values(), out);
    return;
  }

  throw Error("expected list column for embeddings in " + path.string());
}

void append_embeddings_from_parquet(const std::filesystem::path& path, const std::string& column_name,
                                    std::size_t dim, std::vector<float>& out) {
  auto maybe_infile = arrow::io::ReadableFile::Open(path.string());
  if (!maybe_infile.ok()) {
    throw Error("failed to open parquet file " + path.string());
  }
  std::shared_ptr<arrow::io::ReadableFile> infile = *maybe_infile;

  auto maybe_reader = parquet::arrow::OpenFile(infile, arrow::default_memory_pool());
  if (!maybe_reader.ok()) {
    throw Error("failed to read parquet " + path.string());
  }
  std::unique_ptr<parquet::arrow::FileReader> reader = std::move(*maybe_reader);

  auto maybe_table = reader->ReadTable();
  if (!maybe_table.ok()) {
    throw Error("failed to read parquet table " + path.string());
  }
  std::shared_ptr<arrow::Table> table = *maybe_table;

  auto column = table->GetColumnByName(column_name);
  if (!column) {
    throw Error("missing column '" + column_name + "' in " + path.string());
  }

  for (const auto& chunk : column->chunks()) {
    append_embeddings_from_chunk(chunk, dim, path, out);
  }
}

#endif  // VECTORCACHE_FETCH_OPENAI

void fetch_glove(const std::filesystem::path& data_dir, bool force) {
  const auto dest = data_dir / kGloveFilename;
  if (!force) {
    try {
      validate_glove(dest);
      print_dataset_status("GloVe", dest, "train + test HDF5 datasets", true);
      return;
    } catch (...) {
    }
  }

  std::printf("Downloading %s ...\n", kGloveUrl);
  const auto tmp = dest;
  const auto tmp_path = std::filesystem::path(dest.string() + ".tmp");
  download_url_to_file(kGloveUrl, tmp_path, dest);
  validate_glove(dest);
  print_dataset_status("GloVe", dest, "train + test HDF5 datasets", false);
}

#ifdef VECTORCACHE_FETCH_OPENAI
void fetch_openai(std::size_t dim, const std::filesystem::path& data_dir, bool force) {
  const auto filename = "openai-" + std::to_string(dim) + ".npy";
  const auto dest = data_dir / filename;
  if (!force) {
    try {
      validate_openai(dest, dim);
      const auto shape = read_openai_shape(dest);
      print_dataset_status("OpenAI-" + std::to_string(dim), dest,
                           "shape (" + std::to_string(shape.first) + ", " +
                               std::to_string(shape.second) + ")",
                           true);
      return;
    } catch (...) {
    }
  }

  const std::string repo_id = "Qdrant/dbpedia-entities-openai3-text-embedding-3-large-" +
                              std::to_string(dim) + "-1M";
  const std::string column = "text-embedding-3-large-" + std::to_string(dim) + "-embedding";
  const std::size_t shard_count = openai_parquet_shard_count(dim);
  std::printf("Downloading %s (%zu parquet shards) ...\n", repo_id.c_str(), shard_count);

  const auto cache_dir = data_dir / ".cache" / ("openai-" + std::to_string(dim));
  std::filesystem::create_directories(cache_dir);

  std::vector<float> vectors;
  vectors.reserve(1'000'000 * dim);

  for (std::size_t shard = 0; shard < shard_count; ++shard) {
    char shard_buf[64];
    std::snprintf(shard_buf, sizeof(shard_buf), "data/train-%05zu-of-%05zu.parquet", shard,
                  shard_count);
    std::printf("  shard %zu/%zu: %s\n", shard + 1, shard_count, shard_buf);

    const auto file_name = std::filesystem::path(shard_buf).filename();
    const auto local_path = cache_dir / file_name;
    if (!std::filesystem::is_regular_file(local_path)) {
      const std::string url =
          std::string(kHfDatasetBase) + "/" + repo_id + "/resolve/main/" + shard_buf;
      const auto tmp = local_path;
      const auto tmp_path = std::filesystem::path(local_path.string() + ".tmp");
      download_url_to_file(url, tmp_path, local_path);
    }
    append_embeddings_from_parquet(local_path, column, dim, vectors);
  }

  if (vectors.size() / dim != 1'000'000) {
    throw Error("expected 1,000,000 vectors for OpenAI-" + std::to_string(dim));
  }

  const auto tmp_path = std::filesystem::path(dest.string() + ".tmp");
  write_npy_f32_matrix(tmp_path, 1'000'000, dim, vectors.data());
  std::filesystem::rename(tmp_path, dest);
  validate_openai(dest, dim);
  print_dataset_status("OpenAI-" + std::to_string(dim), dest,
                       "shape (1000000, " + std::to_string(dim) + ")", false);
}
#endif  // VECTORCACHE_FETCH_OPENAI

}  // namespace

std::optional<DatasetKind> parse_dataset_kind(const std::string& name) {
  if (name == "glove") return DatasetKind::Glove;
#ifdef VECTORCACHE_FETCH_OPENAI
  if (name == "openai-1536") return DatasetKind::OpenAi1536;
  if (name == "openai-3072") return DatasetKind::OpenAi3072;
#endif
  return std::nullopt;
}

std::vector<DatasetKind> all_dataset_kinds() {
  std::vector<DatasetKind> kinds = {DatasetKind::Glove};
#ifdef VECTORCACHE_FETCH_OPENAI
  kinds.push_back(DatasetKind::OpenAi1536);
  kinds.push_back(DatasetKind::OpenAi3072);
#endif
  return kinds;
}

const char* dataset_label(DatasetKind kind) {
  switch (kind) {
    case DatasetKind::Glove:
      return "glove";
    case DatasetKind::OpenAi1536:
      return "openai-1536";
    case DatasetKind::OpenAi3072:
      return "openai-3072";
  }
  return "";
}

std::size_t dataset_expected_dim(DatasetKind kind) {
  switch (kind) {
    case DatasetKind::Glove:
      return 200;
    case DatasetKind::OpenAi1536:
      return 1536;
    case DatasetKind::OpenAi3072:
      return 3072;
  }
  return 0;
}

std::filesystem::path dataset_path(DatasetKind kind, const std::filesystem::path& data_dir) {
  switch (kind) {
    case DatasetKind::Glove:
      return data_dir / kGloveFilename;
    case DatasetKind::OpenAi1536:
      return data_dir / "openai-1536.npy";
    case DatasetKind::OpenAi3072:
      return data_dir / "openai-3072.npy";
  }
  return data_dir;
}

void fetch(DatasetKind kind, const std::filesystem::path& data_dir, bool force) {
  static bool curl_initialized = false;
  if (!curl_initialized) {
    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl_initialized = true;
  }

  std::filesystem::create_directories(data_dir);
  switch (kind) {
    case DatasetKind::Glove:
      fetch_glove(data_dir, force);
      break;
    case DatasetKind::OpenAi1536:
    case DatasetKind::OpenAi3072:
#ifdef VECTORCACHE_FETCH_OPENAI
      fetch_openai(kind == DatasetKind::OpenAi1536 ? 1536 : 3072, data_dir, force);
#else
      throw Error("OpenAI dataset fetch requires VECTORCACHE_FETCH_OPENAI (Apache Arrow)");
#endif
      break;
  }
}

OpenDatasetReader OpenDatasetReader::open(DatasetKind kind, const std::filesystem::path& data_dir,
                                        DatasetSplit split) {
  OpenDatasetReader reader;
  const auto path = dataset_path(kind, data_dir);

  switch (kind) {
    case DatasetKind::Glove:
#ifdef VECTORCACHE_BUILD_GLOVE
      if (!std::filesystem::is_regular_file(path)) {
        throw Error("GloVe dataset not found: " + path.string());
      }
      reader.kind_ = Kind::Hdf5;
      reader.reader_ = std::make_unique<Hdf5GloveReader>(Hdf5GloveReader::open(path, split));
      return reader;
#else
      throw Error(
          "GloVe ingestion requires VECTORCACHE_BUILD_GLOVE (HDF5 C library must be installed)");
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
