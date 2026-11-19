#pragma once

#include <cstddef>
#include <cstdint>

namespace vectorcache::gpu::detail {

inline constexpr int kGpuTile = 32;
inline constexpr int kScanThreads = 256;
/// Pack this many byte-groups into one uint32 for coalesced vector loads.
inline constexpr int kGroupsPerU32 = 4;
/// Queries fused per scan block (one codes stream, NQ heaps).
/// Shared: NQ query rows (~24 KiB @ d=1536). Centroids live in registers.
inline constexpr int kNqBatch = 4;
/// Fallback fuse width when k is large (register/local pressure).
inline constexpr int kNqBatchLargeK = 2;

struct CudaBuffers {
  float* d_vectors = nullptr;
  /// Layout: [tile][g4][lane] as uint32 (4 consecutive byte-groups packed LE).
  std::uint32_t* d_codes = nullptr;
  float* d_scales = nullptr;
  float* d_residual_norms = nullptr;  // optional; landmark mode
  float* d_centroids = nullptr;
  float* d_boundaries = nullptr;
  std::uint32_t* d_perm = nullptr;
  float* d_signs = nullptr;
  float* d_landmark_basis = nullptr;  // column-major dim * m (search project)
  float* d_queries = nullptr;
  float* d_q_rot = nullptr;   // full ambient dim
  float* d_q_land = nullptr;  // scan_dim (landmark mode)
  float* d_q_perp = nullptr;  // nq residual norms
  float* d_out_scores = nullptr;
  std::uint64_t* d_out_ids = nullptr;

  std::size_t n = 0;
  std::size_t dim = 0;       // ambient dim (rotation)
  std::size_t scan_dim = 0;  // FastScan dim (== dim if no landmarks)
  std::size_t n_landmarks = 0;
  std::size_t bits = 0;
  std::size_t n_cent = 0;
  std::size_t n_byte_groups = 0;
  std::size_t nbg4 = 0;  // ceil(n_byte_groups / 4)
  std::size_t n_tiles = 0;
  std::size_t block = 0;
  float inv_sqrt_block = 0.f;
  bool use_residual = false;
  std::size_t capacity_q = 0;
  std::size_t capacity_k = 0;
};

void cuda_check(int err, const char* what);
void cuda_sync();

void alloc_index_buffers(CudaBuffers& b, std::size_t n, std::size_t ambient_dim,
                         std::size_t scan_dim, std::size_t bits, std::size_t block,
                         float inv_sqrt_block, std::size_t n_landmarks);
void free_buffers(CudaBuffers& b);
void ensure_query_capacity(CudaBuffers& b, std::size_t nq, std::size_t k);

void upload_rotation(CudaBuffers& b, const std::uint32_t* perm, const float* signs);
void upload_codebook(CudaBuffers& b, const float* centroids, const float* boundaries);
void upload_vectors(CudaBuffers& b, const float* host_vectors, std::size_t n);
void upload_residuals(CudaBuffers& b, const float* host_residuals, std::size_t n);
void upload_landmark_basis(CudaBuffers& b, const float* host_L, std::size_t dim, std::size_t m);

/// Full-dim path: L2 + FWHT + quantize + pack (ambient dim == scan_dim).
void encode_on_device(CudaBuffers& b);
/// Landmark path: L2 + quantize + pack over scan_dim (no FWHT).
void encode_prerotated_on_device(CudaBuffers& b);

void search_on_device(CudaBuffers& b, const float* host_queries, std::size_t nq, std::size_t k,
                      float* host_out_scores, std::uint64_t* host_out_ids);

int device_count();
void device_name(char* buf, std::size_t buf_len);

}  // namespace vectorcache::gpu::detail
