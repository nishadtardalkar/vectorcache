#include "vectorcache/datasets/hdf5.hpp"

#ifdef VECTORCACHE_BUILD_GLOVE

#include <algorithm>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/engine.hpp"

namespace vectorcache::datasets {

namespace {

void check_h5(herr_t status, const std::string& context) {
  if (status < 0) {
    throw Error(context);
  }
}

constexpr std::size_t kChunkRows = ingest::INGEST_BATCH_SIZE * 4;

}  // namespace

Hdf5GloveReader::Hdf5GloveReader(hid_t file_id, hid_t dataset_id, std::string dataset_name,
                                 std::size_t dim, std::size_t count)
    : file_id_(file_id),
      dataset_id_(dataset_id),
      dataset_name_(std::move(dataset_name)),
      dim_(dim),
      count_(count),
      index_(0),
      chunk_len_(0),
      chunk_pos_(0) {}

Hdf5GloveReader::~Hdf5GloveReader() { close(); }

Hdf5GloveReader::Hdf5GloveReader(Hdf5GloveReader&& other) noexcept
    : file_id_(other.file_id_),
      dataset_id_(other.dataset_id_),
      dataset_name_(std::move(other.dataset_name_)),
      dim_(other.dim_),
      count_(other.count_),
      index_(other.index_),
      chunk_buffer_(std::move(other.chunk_buffer_)),
      chunk_len_(other.chunk_len_),
      chunk_pos_(other.chunk_pos_) {
  other.file_id_ = H5I_INVALID_HID;
  other.dataset_id_ = H5I_INVALID_HID;
}

Hdf5GloveReader& Hdf5GloveReader::operator=(Hdf5GloveReader&& other) noexcept {
  if (this != &other) {
    close();
    file_id_ = other.file_id_;
    dataset_id_ = other.dataset_id_;
    dataset_name_ = std::move(other.dataset_name_);
    dim_ = other.dim_;
    count_ = other.count_;
    index_ = other.index_;
    chunk_buffer_ = std::move(other.chunk_buffer_);
    chunk_len_ = other.chunk_len_;
    chunk_pos_ = other.chunk_pos_;
    other.file_id_ = H5I_INVALID_HID;
    other.dataset_id_ = H5I_INVALID_HID;
  }
  return *this;
}

void Hdf5GloveReader::close() {
  if (dataset_id_ >= 0 && H5Iis_valid(dataset_id_)) {
    H5Dclose(dataset_id_);
  }
  dataset_id_ = H5I_INVALID_HID;
  if (file_id_ >= 0 && H5Iis_valid(file_id_)) {
    H5Fclose(file_id_);
  }
  file_id_ = H5I_INVALID_HID;
}

Hdf5GloveReader Hdf5GloveReader::open(const std::filesystem::path& path, DatasetSplit split) {
  const hid_t file_id = H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  if (file_id < 0) {
    throw Error("failed to open GloVe HDF5 file: " + path.string());
  }

  const char* dataset_name = hdf5_dataset_name(split);
  const hid_t dataset_id = H5Dopen2(file_id, dataset_name, H5P_DEFAULT);
  if (dataset_id < 0) {
    H5Fclose(file_id);
    throw Error(std::string("missing '") + dataset_name + "' dataset in GloVe HDF5");
  }

  const hid_t space_id = H5Dget_space(dataset_id);
  const int rank = H5Sget_simple_extent_ndims(space_id);
  if (rank != 2) {
    H5Sclose(space_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);
    throw Error("expected 2-D GloVe dataset");
  }

  hsize_t dims[2] = {0, 0};
  check_h5(H5Sget_simple_extent_dims(space_id, dims, nullptr), "failed to read HDF5 shape");
  H5Sclose(space_id);

  return Hdf5GloveReader(file_id, dataset_id, dataset_name, static_cast<std::size_t>(dims[1]),
                         static_cast<std::size_t>(dims[0]));
}

DatasetMeta Hdf5GloveReader::meta() const {
  return DatasetMeta{dim_, count_, "glove"};
}

void Hdf5GloveReader::ensure_chunk() {
  if (chunk_pos_ < chunk_len_) {
    return;
  }
  if (index_ >= count_) {
    return;
  }

  if (dataset_id_ < 0 || !H5Iis_valid(dataset_id_)) {
    throw Error("missing '" + dataset_name_ + "' dataset in GloVe HDF5");
  }

  const std::size_t remaining = count_ - index_;
  const std::size_t chunk_rows = std::min(remaining, kChunkRows);
  const std::size_t buffer_len = chunk_rows * dim_;

  chunk_buffer_.assign(buffer_len, 0.0f);
  chunk_len_ = chunk_rows;
  chunk_pos_ = 0;

  hsize_t offset[2] = {static_cast<hsize_t>(index_), 0};
  hsize_t count[2] = {static_cast<hsize_t>(chunk_rows), static_cast<hsize_t>(dim_)};

  const hid_t file_space = H5Dget_space(dataset_id_);
  check_h5(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, offset, nullptr, count, nullptr),
           "failed to select HDF5 hyperslab");

  const hid_t mem_space = H5Screate_simple(2, count, nullptr);
  const hid_t xfer = H5Pcreate(H5P_DATASET_XFER);
  check_h5(H5Dread(dataset_id_, H5T_IEEE_F32LE, mem_space, file_space, xfer, chunk_buffer_.data()),
           "failed to read HDF5 rows");

  H5Pclose(xfer);
  H5Sclose(mem_space);
  H5Sclose(file_space);
}

bool Hdf5GloveReader::next_vector_into(std::span<float> out) {
  if (index_ >= count_) {
    return false;
  }
  if (out.size() != dim_) {
    throw Error("buffer dimension mismatch: expected " + std::to_string(dim_) + ", got " +
                std::to_string(out.size()));
  }

  ensure_chunk();

  const std::size_t offset = chunk_pos_ * dim_;
  std::copy(chunk_buffer_.begin() + static_cast<std::ptrdiff_t>(offset),
            chunk_buffer_.begin() + static_cast<std::ptrdiff_t>(offset + dim_), out.begin());
  ++chunk_pos_;
  ++index_;
  return true;
}

}  // namespace vectorcache::datasets

#endif
