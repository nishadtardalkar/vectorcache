#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace vectorcache::gpu::detail {
namespace {

constexpr float kMinNorm = 1e-10f;
constexpr double kDegenerateInner = 1e-12;
constexpr int kMaxK = 256;

#define VC_CUDA(expr) cuda_check(static_cast<int>(expr), #expr)

__device__ void wht_shared(float* data, int n, float inv_sqrt, int tid, int nthreads) {
  for (int len = 1; len < n; len <<= 1) {
    for (int i = tid; i < n / 2; i += nthreads) {
      const int mid = (i / len) * (2 * len) + (i % len);
      const int j = mid + len;
      const float a = data[mid];
      const float b = data[j];
      data[mid] = a + b;
      data[j] = a - b;
    }
    __syncthreads();
  }
  for (int i = tid; i < n; i += nthreads) data[i] *= inv_sqrt;
  __syncthreads();
}

__device__ inline unsigned quantize_code(float x, const float* boundaries, int n_bounds) {
  unsigned code = 0;
#pragma unroll 8
  for (int bi = 0; bi < n_bounds; ++bi) code += static_cast<unsigned>(x > boundaries[bi]);
  return code;
}

__device__ float block_reduce_sum(float local, int tid) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    local += __shfl_down_sync(0xffffffffu, local, offset);
  }
  __shared__ float red[32];
  if ((tid & 31) == 0) red[tid >> 5] = local;
  __syncthreads();
  if (tid < 32) {
    float v = (tid < ((blockDim.x + 31) >> 5)) ? red[tid] : 0.f;
    for (int offset = 16; offset > 0; offset >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    if (tid == 0) red[0] = v;
  }
  __syncthreads();
  return red[0];
}

__global__ void encode_coop(const float* __restrict__ vectors, std::uint8_t* __restrict__ codes_tiled,
                            float* __restrict__ scales, const float* __restrict__ centroids,
                            const float* __restrict__ boundaries, const unsigned* __restrict__ perm,
                            const float* __restrict__ signs, int n, int dim, int bits, int n_cent,
                            int n_bounds, int nbg, int wht_block, float inv_sqrt) {
  const int vi = blockIdx.x;
  if (vi >= n) return;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* buf = smem;
  const float* src = vectors + static_cast<std::size_t>(vi) * dim;

  float local = 0.f;
  for (int j = tid; j < dim; j += blockDim.x) {
    const float x = __ldg(src + j);
    local += x * x;
  }
  const float norm = sqrtf(block_reduce_sum(local, tid));
  const float inv = (norm > kMinNorm) ? (1.f / norm) : 0.f;

  for (int j = tid; j < dim; j += blockDim.x) {
    buf[j] = (__ldg(src + __ldg(perm + j)) * inv) * __ldg(signs + j);
  }
  __syncthreads();
  for (int off = 0; off < dim; off += wht_block) {
    wht_shared(buf + off, wht_block, inv_sqrt, tid, blockDim.x);
  }

  float partial_inner = 0.f;
  for (int j = tid; j < dim; j += blockDim.x) {
    const float x = buf[j];
    unsigned code = quantize_code(x, boundaries, n_bounds);
    if (static_cast<int>(code) >= n_cent) code = static_cast<unsigned>(n_cent - 1);
    partial_inner += x * centroids[code];
    buf[j] = __int_as_float(static_cast<int>(code));
  }
  const float inner = block_reduce_sum(partial_inner, tid);
  if (tid == 0) {
    scales[vi] = (static_cast<double>(inner) > kDegenerateInner) ? (norm / inner) : 0.f;
  }
  __syncthreads();

  const int tile = vi / kGpuTile;
  const int lane = vi - tile * kGpuTile;
  if (bits == 4 || bits == 3) {
    for (int g = tid; g < nbg; g += blockDim.x) {
      const unsigned c0 = static_cast<unsigned>(__float_as_int(buf[2 * g]));
      const unsigned c1 = static_cast<unsigned>(__float_as_int(buf[2 * g + 1]));
      codes_tiled[(static_cast<std::size_t>(tile) * nbg + g) * kGpuTile + lane] =
          static_cast<std::uint8_t>(((c0 & 0xfu) << 4) | (c1 & 0xfu));
    }
  } else {
    for (int g = tid; g < nbg; g += blockDim.x) {
      const unsigned c0 = static_cast<unsigned>(__float_as_int(buf[4 * g])) & 3u;
      const unsigned c1 = static_cast<unsigned>(__float_as_int(buf[4 * g + 1])) & 3u;
      const unsigned c2 = static_cast<unsigned>(__float_as_int(buf[4 * g + 2])) & 3u;
      const unsigned c3 = static_cast<unsigned>(__float_as_int(buf[4 * g + 3])) & 3u;
      codes_tiled[(static_cast<std::size_t>(tile) * nbg + g) * kGpuTile + lane] =
          static_cast<std::uint8_t>((((c0 << 2) | c1) << 4) | ((c2 << 2) | c3));
    }
  }
}

__global__ void rotate_query_coop(const float* __restrict__ queries, float* __restrict__ q_rot,
                                  const unsigned* __restrict__ perm, const float* __restrict__ signs,
                                  int nq, int dim, int wht_block, float inv_sqrt) {
  const int qi = blockIdx.x;
  if (qi >= nq) return;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* buf = smem;
  const float* src = queries + static_cast<std::size_t>(qi) * dim;
  float* dst = q_rot + static_cast<std::size_t>(qi) * dim;

  float local = 0.f;
  for (int j = tid; j < dim; j += blockDim.x) {
    const float x = __ldg(src + j);
    local += x * x;
  }
  const float norm_sq = block_reduce_sum(local, tid);
  const float inv = (norm_sq > kMinNorm * kMinNorm) ? rsqrtf(norm_sq) : 0.f;

  for (int j = tid; j < dim; j += blockDim.x) {
    buf[j] = (__ldg(src + __ldg(perm + j)) * inv) * __ldg(signs + j);
  }
  __syncthreads();
  for (int off = 0; off < dim; off += wht_block) {
    wht_shared(buf + off, wht_block, inv_sqrt, tid, blockDim.x);
  }
  for (int j = tid; j < dim; j += blockDim.x) dst[j] = buf[j];
}

__device__ inline float score_vector_4bit(const std::uint8_t* __restrict__ base,
                                         const float* __restrict__ sq, const float* __restrict__ sc,
                                         int nbg) {
  float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
  int g = 0;
  for (; g + 3 < nbg; g += 4) {
    const unsigned b0 = base[g * kGpuTile];
    const unsigned b1 = base[(g + 1) * kGpuTile];
    const unsigned b2 = base[(g + 2) * kGpuTile];
    const unsigned b3 = base[(g + 3) * kGpuTile];
    a0 += sq[2 * g] * sc[b0 >> 4] + sq[2 * g + 1] * sc[b0 & 0xfu];
    a1 += sq[2 * (g + 1)] * sc[b1 >> 4] + sq[2 * (g + 1) + 1] * sc[b1 & 0xfu];
    a2 += sq[2 * (g + 2)] * sc[b2 >> 4] + sq[2 * (g + 2) + 1] * sc[b2 & 0xfu];
    a3 += sq[2 * (g + 3)] * sc[b3 >> 4] + sq[2 * (g + 3) + 1] * sc[b3 & 0xfu];
  }
  for (; g < nbg; ++g) {
    const unsigned b = base[g * kGpuTile];
    a0 += sq[2 * g] * sc[b >> 4] + sq[2 * g + 1] * sc[b & 0xfu];
  }
  return (a0 + a1) + (a2 + a3);
}

__device__ inline float score_vector_2bit(const std::uint8_t* __restrict__ base,
                                         const float* __restrict__ sq, const float* __restrict__ sc,
                                         int nbg) {
  float a0 = 0.f, a1 = 0.f;
  for (int g = 0; g < nbg; ++g) {
    const unsigned b = base[g * kGpuTile];
    const unsigned lo = b >> 4;
    const unsigned hi = b & 0xfu;
    a0 += sq[4 * g] * sc[(lo >> 2) & 3u] + sq[4 * g + 1] * sc[lo & 3u];
    a1 += sq[4 * g + 2] * sc[(hi >> 2) & 3u] + sq[4 * g + 3] * sc[hi & 3u];
  }
  return a0 + a1;
}

__device__ inline void heap_push_min(float* hs, int* hi, int& sz, int k, float score, int id) {
  if (sz < k) {
    int i = sz++;
    hs[i] = score;
    hi[i] = id;
    while (i > 0) {
      const int p = (i - 1) >> 1;
      if (hs[p] <= hs[i]) break;
      const float ts = hs[p];
      const int ti = hi[p];
      hs[p] = hs[i];
      hi[p] = hi[i];
      hs[i] = ts;
      hi[i] = ti;
      i = p;
    }
    return;
  }
  if (score < hs[0] || (score == hs[0] && id >= hi[0])) return;
  hs[0] = score;
  hi[0] = id;
  int i = 0;
  for (;;) {
    const int l = (i << 1) + 1;
    const int r = l + 1;
    if (l >= sz) break;
    int m = l;
    if (r < sz && (hs[r] < hs[l] || (hs[r] == hs[l] && hi[r] < hi[l]))) m = r;
    if (hs[i] < hs[m] || (hs[i] == hs[m] && hi[i] <= hi[m])) break;
    const float ts = hs[i];
    const int ti = hi[i];
    hs[i] = hs[m];
    hi[i] = hi[m];
    hs[m] = ts;
    hi[m] = ti;
    i = m;
  }
}

__device__ void sort_desc(float* s, int* ids, int n) {
  for (int i = 0; i < n; ++i) {
    int best = i;
    for (int j = i + 1; j < n; ++j) {
      if (s[j] > s[best] || (s[j] == s[best] && ids[j] < ids[best])) best = j;
    }
    const float ts = s[i];
    const int ti = ids[i];
    s[i] = s[best];
    ids[i] = ids[best];
    s[best] = ts;
    ids[best] = ti;
  }
}

template <int kCap>
__global__ void __launch_bounds__(kScanThreads, 4)
    scan_fused_topk_k(const std::uint8_t* __restrict__ codes_tiled, const float* __restrict__ scales,
                      const float* __restrict__ q_rot, const float* __restrict__ centroids,
                      float* __restrict__ out_scores, unsigned long long* __restrict__ out_ids, int n,
                      int dim, int bits, int n_cent, int nbg, int k) {
  const int qi = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* sq = smem;
  float* sc = sq + dim;

  for (int j = tid; j < dim; j += blockDim.x) {
    sq[j] = q_rot[static_cast<std::size_t>(qi) * dim + j];
  }
  for (int j = tid; j < n_cent; j += blockDim.x) sc[j] = centroids[j];
  __syncthreads();

  float loc_s[kCap];
  int loc_i[kCap];
  int loc_sz = 0;
  const int kk = (k < kCap) ? k : kCap;

  const int n_tiles = (n + kGpuTile - 1) / kGpuTile;
  const int tile_stride = blockDim.x / kGpuTile;
  for (int tile = tid / kGpuTile; tile < n_tiles; tile += tile_stride) {
    const int lane = tid & (kGpuTile - 1);
    const int vi = tile * kGpuTile + lane;
    if (vi >= n) continue;
    const std::uint8_t* base =
        codes_tiled + static_cast<std::size_t>(tile) * nbg * kGpuTile + lane;
    float s = (bits == 2) ? score_vector_2bit(base, sq, sc, nbg)
                          : score_vector_4bit(base, sq, sc, nbg);
    s *= __ldg(scales + vi);
    heap_push_min(loc_s, loc_i, loc_sz, kk, s, vi);
  }

  __shared__ float warp_s[kScanThreads / 32][kCap];
  __shared__ int warp_i[kScanThreads / 32][kCap];
  __shared__ int warp_n[kScanThreads / 32];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) warp_n[warp] = 0;
  __syncwarp();
  for (int t = 0; t < 32; ++t) {
    if (lane == t) {
      for (int i = 0; i < loc_sz; ++i) {
        heap_push_min(warp_s[warp], warp_i[warp], warp_n[warp], kk, loc_s[i], loc_i[i]);
      }
    }
    __syncwarp();
  }
  __syncthreads();

  if (tid == 0) {
    float blk_s[kCap];
    int blk_i[kCap];
    int blk_n = 0;
    const int nwarps = blockDim.x >> 5;
    for (int w = 0; w < nwarps; ++w) {
      for (int i = 0; i < warp_n[w]; ++i) {
        heap_push_min(blk_s, blk_i, blk_n, kk, warp_s[w][i], warp_i[w][i]);
      }
    }
    sort_desc(blk_s, blk_i, blk_n);
    float* os = out_scores + static_cast<std::size_t>(qi) * k;
    unsigned long long* oi = out_ids + static_cast<std::size_t>(qi) * k;
    for (int i = 0; i < k; ++i) {
      if (i < blk_n) {
        os[i] = blk_s[i];
        oi[i] = static_cast<unsigned long long>(blk_i[i]);
      } else {
        os[i] = -1e30f;
        oi[i] = 0;
      }
    }
  }
}

}  // namespace

void cuda_check(int err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string("CUDA error in ") + what + ": " +
                             cudaGetErrorString(static_cast<cudaError_t>(err)));
  }
}

void cuda_sync() { VC_CUDA(cudaDeviceSynchronize()); }

int device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

void device_name(char* buf, std::size_t buf_len) {
  if (!buf || buf_len == 0) return;
  buf[0] = '\0';
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
    std::snprintf(buf, buf_len, "%s", prop.name);
  }
}

void free_buffers(CudaBuffers& b) {
  auto free_p = [](auto*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_p(b.d_vectors);
  free_p(b.d_codes);
  free_p(b.d_scales);
  free_p(b.d_centroids);
  free_p(b.d_boundaries);
  free_p(b.d_perm);
  free_p(b.d_signs);
  free_p(b.d_queries);
  free_p(b.d_q_rot);
  free_p(b.d_scores);
  free_p(b.d_out_scores);
  free_p(b.d_out_ids);
  b.capacity_q = 0;
  b.capacity_k = 0;
}

void alloc_index_buffers(CudaBuffers& b, std::size_t n, std::size_t dim, std::size_t bits,
                         std::size_t block, float inv_sqrt_block) {
  free_buffers(b);
  b.n = n;
  b.n_pad = 0;
  b.dim = dim;
  b.bits = bits;
  b.n_cent = 1u << bits;
  b.n_byte_groups = dim / (8 / bits);
  b.n_tiles = (n + kGpuTile - 1) / kGpuTile;
  b.block = block;
  b.inv_sqrt_block = inv_sqrt_block;

  const std::size_t codes_bytes = b.n_tiles * b.n_byte_groups * kGpuTile;
  VC_CUDA(cudaMalloc(&b.d_vectors, n * dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_codes, codes_bytes));
  VC_CUDA(cudaMalloc(&b.d_scales, n * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_centroids, b.n_cent * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_boundaries, (b.n_cent > 1 ? b.n_cent - 1 : 1) * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_perm, dim * sizeof(std::uint32_t)));
  VC_CUDA(cudaMalloc(&b.d_signs, dim * sizeof(float)));
  VC_CUDA(cudaMemset(b.d_codes, 0, codes_bytes));
}

void ensure_query_capacity(CudaBuffers& b, std::size_t nq, std::size_t k) {
  if (nq <= b.capacity_q && k <= b.capacity_k && b.d_queries) return;
  auto free_p = [](auto*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_p(b.d_queries);
  free_p(b.d_q_rot);
  free_p(b.d_out_scores);
  free_p(b.d_out_ids);

  b.capacity_q = std::max(nq, std::size_t{1});
  b.capacity_k = std::max(k, std::size_t{1});
  VC_CUDA(cudaMalloc(&b.d_queries, b.capacity_q * b.dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_q_rot, b.capacity_q * b.dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_out_scores, b.capacity_q * b.capacity_k * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_out_ids, b.capacity_q * b.capacity_k * sizeof(std::uint64_t)));
}

void upload_rotation(CudaBuffers& b, const std::uint32_t* perm, const float* signs) {
  VC_CUDA(cudaMemcpy(b.d_perm, perm, b.dim * sizeof(std::uint32_t), cudaMemcpyHostToDevice));
  VC_CUDA(cudaMemcpy(b.d_signs, signs, b.dim * sizeof(float), cudaMemcpyHostToDevice));
}

void upload_codebook(CudaBuffers& b, const float* centroids, const float* boundaries) {
  VC_CUDA(cudaMemcpy(b.d_centroids, centroids, b.n_cent * sizeof(float), cudaMemcpyHostToDevice));
  if (b.n_cent > 1) {
    VC_CUDA(cudaMemcpy(b.d_boundaries, boundaries, (b.n_cent - 1) * sizeof(float),
                       cudaMemcpyHostToDevice));
  }
}

void upload_vectors(CudaBuffers& b, const float* host_vectors, std::size_t n) {
  VC_CUDA(cudaMemcpy(b.d_vectors, host_vectors, n * b.dim * sizeof(float), cudaMemcpyHostToDevice));
}

void encode_on_device(CudaBuffers& b) {
  encode_coop<<<static_cast<int>(b.n), 256, b.dim * sizeof(float)>>>(
      b.d_vectors, b.d_codes, b.d_scales, b.d_centroids, b.d_boundaries, b.d_perm, b.d_signs,
      static_cast<int>(b.n), static_cast<int>(b.dim), static_cast<int>(b.bits),
      static_cast<int>(b.n_cent), static_cast<int>(b.n_cent > 1 ? b.n_cent - 1 : 0),
      static_cast<int>(b.n_byte_groups), static_cast<int>(b.block), b.inv_sqrt_block);
  VC_CUDA(cudaGetLastError());
  cuda_sync();
}

void search_on_device(CudaBuffers& b, const float* host_queries, std::size_t nq, std::size_t k,
                      float* host_out_scores, std::uint64_t* host_out_ids) {
  if (k == 0 || nq == 0) return;
  if (k > kMaxK) throw std::invalid_argument("GPU search k must be <= 256");
  ensure_query_capacity(b, nq, k);
  VC_CUDA(cudaMemcpy(b.d_queries, host_queries, nq * b.dim * sizeof(float), cudaMemcpyHostToDevice));

  rotate_query_coop<<<static_cast<int>(nq), 256, b.dim * sizeof(float)>>>(
      b.d_queries, b.d_q_rot, b.d_perm, b.d_signs, static_cast<int>(nq), static_cast<int>(b.dim),
      static_cast<int>(b.block), b.inv_sqrt_block);
  VC_CUDA(cudaGetLastError());

  {
    const std::size_t shmem = (b.dim + b.n_cent) * sizeof(float);
    const int nqi = static_cast<int>(nq);
    const int ni = static_cast<int>(b.n);
    const int di = static_cast<int>(b.dim);
    const int bi = static_cast<int>(b.bits);
    const int nc = static_cast<int>(b.n_cent);
    const int nbg = static_cast<int>(b.n_byte_groups);
    const int ki = static_cast<int>(k);
    if (k <= 64) {
      scan_fused_topk_k<64><<<nqi, kScanThreads, shmem>>>(
          b.d_codes, b.d_scales, b.d_q_rot, b.d_centroids, b.d_out_scores, b.d_out_ids, ni, di, bi,
          nc, nbg, ki);
    } else {
      scan_fused_topk_k<256><<<nqi, kScanThreads, shmem>>>(
          b.d_codes, b.d_scales, b.d_q_rot, b.d_centroids, b.d_out_scores, b.d_out_ids, ni, di, bi,
          nc, nbg, ki);
    }
    VC_CUDA(cudaGetLastError());
  }

  cuda_sync();
  VC_CUDA(cudaMemcpy(host_out_scores, b.d_out_scores, nq * k * sizeof(float),
                     cudaMemcpyDeviceToHost));
  VC_CUDA(cudaMemcpy(host_out_ids, b.d_out_ids, nq * k * sizeof(std::uint64_t),
                     cudaMemcpyDeviceToHost));
}

}  // namespace vectorcache::gpu::detail
