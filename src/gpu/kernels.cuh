#pragma once

#include <cstddef>
#include <cstdint>

namespace vectorcache::gpu::detail {

inline constexpr int kGpuTile = 32;
inline constexpr int kScanThreads = 256;

struct CudaBuffers {
  float* d_vectors = nullptr;
  std::uint8_t* d_codes = nullptr;
  float* d_scales = nullptr;
  float* d_centroids = nullptr;
  float* d_boundaries = nullptr;
  std::uint32_t* d_perm = nullptr;
  float* d_signs = nullptr;
  float* d_queries = nullptr;
  float* d_q_rot = nullptr;
  float* d_scores = nullptr;  // nq * n_pad score buffer
  float* d_out_scores = nullptr;
  std::uint64_t* d_out_ids = nullptr;

  std::size_t n = 0;
  std::size_t n_pad = 0;  // padded to kScanThreads
  std::size_t dim = 0;
  std::size_t bits = 0;
  std::size_t n_cent = 0;
  std::size_t n_byte_groups = 0;
  std::size_t n_tiles = 0;
  std::size_t block = 0;
  float inv_sqrt_block = 0.f;
  std::size_t capacity_q = 0;
  std::size_t capacity_k = 0;
};

void cuda_check(int err, const char* what);
void cuda_sync();

void alloc_index_buffers(CudaBuffers& b, std::size_t n, std::size_t dim, std::size_t bits,
                         std::size_t block, float inv_sqrt_block);
void free_buffers(CudaBuffers& b);
void ensure_query_capacity(CudaBuffers& b, std::size_t nq, std::size_t k);

void upload_rotation(CudaBuffers& b, const std::uint32_t* perm, const float* signs);
void upload_codebook(CudaBuffers& b, const float* centroids, const float* boundaries);
void upload_vectors(CudaBuffers& b, const float* host_vectors, std::size_t n);

void encode_on_device(CudaBuffers& b);
void search_on_device(CudaBuffers& b, const float* host_queries, std::size_t nq, std::size_t k,
                      float* host_out_scores, std::uint64_t* host_out_ids);

int device_count();
void device_name(char* buf, std::size_t buf_len);

}  // namespace vectorcache::gpu::detail
