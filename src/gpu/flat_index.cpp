#include "vectorcache/gpu/flat_index.hpp"

#include <stdexcept>
#include <utility>

#include "kernels.cuh"
#include "vectorcache/constants.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache::gpu {

struct FlatIndex::DeviceState {
  detail::CudaBuffers buf;
  Rotation rotation;
  explicit DeviceState(std::size_t dim) : rotation(dim) {}
};

bool cuda_available() { return detail::device_count() > 0; }

FlatIndex::FlatIndex(std::size_t dim, std::size_t bits) : dim_(dim), bits_(bits) {
  if (dim == 0 || dim % 8 != 0 || dim > kMaxDim) {
    throw std::invalid_argument("GpuFlatIndex dim must be a positive multiple of 8, <= 16384");
  }
  if (bits < 2 || bits > 4) {
    throw std::invalid_argument("GpuFlatIndex bits must be 2, 3, or 4");
  }
  if (!cuda_available()) {
    throw std::runtime_error("CUDA device not available");
  }
  char name[256];
  detail::device_name(name, sizeof(name));
  device_name_ = name;

  auto cb = codebook(bits_, dim_);
  boundaries_ = std::move(cb.first);
  centroids_ = std::move(cb.second);
  dev_ = std::make_unique<DeviceState>(dim_);
}

FlatIndex::~FlatIndex() = default;
FlatIndex::FlatIndex(FlatIndex&&) noexcept = default;
FlatIndex& FlatIndex::operator=(FlatIndex&&) noexcept = default;

void FlatIndex::add(std::span<const float> vectors) {
  if (vectors.empty()) return;
  if (vectors.size() % dim_ != 0) {
    throw std::invalid_argument("add: vectors size must be multiple of dim");
  }
  if (prepared_) {
    throw std::runtime_error("add after prepare is not supported for GpuFlatIndex");
  }
  const std::size_t n = vectors.size() / dim_;
  host_vectors_.insert(host_vectors_.end(), vectors.begin(), vectors.end());
  n_vectors_ += n;
}

void FlatIndex::prepare() {
  if (prepared_) return;
  if (n_vectors_ == 0) {
    throw std::runtime_error("GpuFlatIndex::prepare with empty index");
  }
  auto& b = dev_->buf;
  detail::alloc_index_buffers(b, n_vectors_, dim_, bits_, dev_->rotation.block(),
                              dev_->rotation.inv_sqrt_block());
  detail::upload_rotation(b, dev_->rotation.perm(0).data(), dev_->rotation.signs(0).data());
  detail::upload_codebook(b, centroids_.data(), boundaries_.data());
  detail::upload_vectors(b, host_vectors_.data(), n_vectors_);
  detail::encode_on_device(b);
  host_vectors_.clear();
  host_vectors_.shrink_to_fit();
  prepared_ = true;
}

SearchResults FlatIndex::search(std::span<const float> queries, std::size_t k) const {
  if (!prepared_) {
    throw std::runtime_error("GpuFlatIndex::search before prepare");
  }
  if (queries.empty()) {
    SearchResults empty;
    empty.nq = 0;
    empty.k = k;
    return empty;
  }
  if (queries.size() % dim_ != 0) {
    throw std::invalid_argument("search: queries size must be multiple of dim");
  }
  const std::size_t nq = queries.size() / dim_;
  const std::size_t kk = std::min(k, n_vectors_);
  SearchResults out;
  out.nq = nq;
  out.k = kk;
  out.scores.resize_uninitialized(nq * kk);
  out.ids.resize_uninitialized(nq * kk);
  detail::search_on_device(dev_->buf, queries.data(), nq, kk, out.scores.data(), out.ids.data());
  return out;
}

}  // namespace vectorcache::gpu
