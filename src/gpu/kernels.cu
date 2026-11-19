#include "kernels.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

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

__device__ inline unsigned pack_byte_4bit(unsigned c0, unsigned c1) {
  return ((c0 & 0xfu) << 4) | (c1 & 0xfu);
}

__device__ inline unsigned pack_byte_2bit(unsigned c0, unsigned c1, unsigned c2, unsigned c3) {
  return ((((c0 & 3u) << 2) | (c1 & 3u)) << 4) | (((c2 & 3u) << 2) | (c3 & 3u));
}

__global__ void encode_coop(const float* __restrict__ vectors, std::uint32_t* __restrict__ codes_u32,
                            float* __restrict__ scales, const float* __restrict__ centroids,
                            const float* __restrict__ boundaries, const unsigned* __restrict__ perm,
                            const float* __restrict__ signs, int n, int dim, int bits, int n_cent,
                            int n_bounds, int nbg, int nbg4, int wht_block, float inv_sqrt) {
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
  // Pack 4 byte-groups into one LE uint32: [b0 | b1<<8 | b2<<16 | b3<<24]
  for (int g4 = tid; g4 < nbg4; g4 += blockDim.x) {
    unsigned bytes[4] = {0, 0, 0, 0};
    for (int t = 0; t < kGroupsPerU32; ++t) {
      const int g = g4 * kGroupsPerU32 + t;
      if (g >= nbg) break;
      if (bits == 2) {
        const unsigned c0 = static_cast<unsigned>(__float_as_int(buf[4 * g]));
        const unsigned c1 = static_cast<unsigned>(__float_as_int(buf[4 * g + 1]));
        const unsigned c2 = static_cast<unsigned>(__float_as_int(buf[4 * g + 2]));
        const unsigned c3 = static_cast<unsigned>(__float_as_int(buf[4 * g + 3]));
        bytes[t] = pack_byte_2bit(c0, c1, c2, c3);
      } else {
        const unsigned c0 = static_cast<unsigned>(__float_as_int(buf[2 * g]));
        const unsigned c1 = static_cast<unsigned>(__float_as_int(buf[2 * g + 1]));
        bytes[t] = pack_byte_4bit(c0, c1);
      }
    }
    const unsigned packed = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    codes_u32[(static_cast<std::size_t>(tile) * nbg4 + g4) * kGpuTile + lane] = packed;
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

/// Score NQ queries from one codes stream (codes loaded once; q in shared, centroids in regs).
template <int NQ>
__device__ inline void score_mq_u32_4bit(const std::uint32_t* __restrict__ base_u32,
                                        const float* __restrict__ sq, int dim,
                                        const float* __restrict__ cent, int nbg4, float out[NQ]) {
  float a[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) a[qi] = 0.f;
  for (int g4 = 0; g4 + 1 < nbg4; g4 += 2) {
    const unsigned p0 = __ldg(base_u32 + g4 * kGpuTile);
    const unsigned p1 = __ldg(base_u32 + (g4 + 1) * kGpuTile);
    const int g = g4 * kGroupsPerU32;
    const unsigned b0 = p0 & 0xffu, b1 = (p0 >> 8) & 0xffu, b2 = (p0 >> 16) & 0xffu,
                   b3 = (p0 >> 24) & 0xffu;
    const unsigned b4 = p1 & 0xffu, b5 = (p1 >> 8) & 0xffu, b6 = (p1 >> 16) & 0xffu,
                   b7 = (p1 >> 24) & 0xffu;
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
      const float* q = sq + static_cast<std::size_t>(qi) * dim;
      a[qi] += q[2 * g] * cent[b0 >> 4] + q[2 * g + 1] * cent[b0 & 0xfu];
      a[qi] += q[2 * (g + 1)] * cent[b1 >> 4] + q[2 * (g + 1) + 1] * cent[b1 & 0xfu];
      a[qi] += q[2 * (g + 2)] * cent[b2 >> 4] + q[2 * (g + 2) + 1] * cent[b2 & 0xfu];
      a[qi] += q[2 * (g + 3)] * cent[b3 >> 4] + q[2 * (g + 3) + 1] * cent[b3 & 0xfu];
      a[qi] += q[2 * (g + 4)] * cent[b4 >> 4] + q[2 * (g + 4) + 1] * cent[b4 & 0xfu];
      a[qi] += q[2 * (g + 5)] * cent[b5 >> 4] + q[2 * (g + 5) + 1] * cent[b5 & 0xfu];
      a[qi] += q[2 * (g + 6)] * cent[b6 >> 4] + q[2 * (g + 6) + 1] * cent[b6 & 0xfu];
      a[qi] += q[2 * (g + 7)] * cent[b7 >> 4] + q[2 * (g + 7) + 1] * cent[b7 & 0xfu];
    }
  }
  if (nbg4 & 1) {
    const int g4 = nbg4 - 1;
    const unsigned p = __ldg(base_u32 + g4 * kGpuTile);
    const int g = g4 * kGroupsPerU32;
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
      const float* q = sq + static_cast<std::size_t>(qi) * dim;
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const unsigned byte = (p >> (8 * bi)) & 0xffu;
        const int j = 2 * (g + bi);
        a[qi] += q[j] * cent[byte >> 4] + q[j + 1] * cent[byte & 0xfu];
      }
    }
  }
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) out[qi] = a[qi];
}

template <int NQ>
__device__ inline void score_mq_u32_2bit(const std::uint32_t* __restrict__ base_u32,
                                        const float* __restrict__ sq, int dim,
                                        const float* __restrict__ cent, int nbg4, float out[NQ]) {
  float a[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) a[qi] = 0.f;
  for (int g4 = 0; g4 < nbg4; ++g4) {
    const unsigned p = __ldg(base_u32 + g4 * kGpuTile);
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
      const float* q = sq + static_cast<std::size_t>(qi) * dim;
      const int g = g4 * kGroupsPerU32;
#pragma unroll
      for (int bi = 0; bi < 4; ++bi) {
        const unsigned byte = (p >> (8 * bi)) & 0xffu;
        const unsigned lo = byte >> 4;
        const unsigned hi = byte & 0xfu;
        const int j = 4 * (g + bi);
        a[qi] += q[j] * cent[(lo >> 2) & 3u] + q[j + 1] * cent[lo & 3u] +
                 q[j + 2] * cent[(hi >> 2) & 3u] + q[j + 3] * cent[hi & 3u];
      }
    }
  }
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) out[qi] = a[qi];
}

__global__ void encode_prerotated_coop(const float* __restrict__ vectors,
                                      std::uint32_t* __restrict__ codes_u32,
                                      float* __restrict__ scales, const float* __restrict__ centroids,
                                      const float* __restrict__ boundaries, int n, int dim, int bits,
                                      int n_cent, int n_bounds, int nbg, int nbg4) {
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

  for (int j = tid; j < dim; j += blockDim.x) buf[j] = __ldg(src + j) * inv;
  __syncthreads();

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
  for (int g4 = tid; g4 < nbg4; g4 += blockDim.x) {
    unsigned bytes[4] = {0, 0, 0, 0};
    for (int t = 0; t < kGroupsPerU32; ++t) {
      const int g = g4 * kGroupsPerU32 + t;
      if (g >= nbg) break;
      if (bits == 2) {
        bytes[t] = pack_byte_2bit(static_cast<unsigned>(__float_as_int(buf[4 * g])),
                                  static_cast<unsigned>(__float_as_int(buf[4 * g + 1])),
                                  static_cast<unsigned>(__float_as_int(buf[4 * g + 2])),
                                  static_cast<unsigned>(__float_as_int(buf[4 * g + 3])));
      } else {
        bytes[t] = pack_byte_4bit(static_cast<unsigned>(__float_as_int(buf[2 * g])),
                                  static_cast<unsigned>(__float_as_int(buf[2 * g + 1])));
      }
    }
    const unsigned packed = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
    codes_u32[(static_cast<std::size_t>(tile) * nbg4 + g4) * kGpuTile + lane] = packed;
  }
}

/// q_land = q_rot @ L (L column-major dim×m); q_perp = sqrt(max(0, ‖q_rot‖² - ‖q_L‖²)).
__global__ void project_queries_landmarks(const float* __restrict__ q_rot,
                                          const float* __restrict__ L, float* __restrict__ q_land,
                                          float* __restrict__ q_perp, int nq, int dim, int m,
                                          int scan_dim) {
  const int qi = blockIdx.x;
  if (qi >= nq) return;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* sq = smem;
  const float* q = q_rot + static_cast<std::size_t>(qi) * dim;
  float* out = q_land + static_cast<std::size_t>(qi) * scan_dim;

  float qn = 0.f;
  for (int d = tid; d < dim; d += blockDim.x) {
    const float v = __ldg(q + d);
    sq[d] = v;
    qn += v * v;
  }
  qn = block_reduce_sum(qn, tid);
  __syncthreads();

  float land_n = 0.f;
  for (int j = tid; j < scan_dim; j += blockDim.x) {
    float s = 0.f;
    if (j < m) {
      const float* col = L + static_cast<std::size_t>(j) * dim;
      for (int d = 0; d < dim; ++d) s += sq[d] * __ldg(col + d);
      land_n += s * s;
    }
    out[j] = s;
  }
  land_n = block_reduce_sum(land_n, tid);
  if (tid == 0) {
    const float r2 = qn - land_n;
    q_perp[qi] = (r2 > 0.f) ? sqrtf(r2) : 0.f;
  }
}

/// Multi-query fused scan: shared queries + register centroids, codes loaded once per vector.
/// Optional residual: score = α·s_L + ‖q_⊥‖·‖r_i‖ (CPU LandmarkIndex parity).
template <int NQ, int kCap, int kLocal, int Bits>
__global__ void __launch_bounds__(kScanThreads, 2)
    scan_mq_topk(const std::uint32_t* __restrict__ codes_u32, const float* __restrict__ scales,
                 const float* __restrict__ residual_norms, const float* __restrict__ q_scan,
                 const float* __restrict__ q_perp, const float* __restrict__ centroids,
                 float* __restrict__ out_scores, unsigned long long* __restrict__ out_ids, int n,
                 int nq_total, int dim, int n_cent, int nbg4, int k, int use_residual) {
  static_assert(kLocal <= kCap, "kLocal");
  const int q_base = blockIdx.x * NQ;
  if (q_base >= nq_total) return;
  const int nq_local = (q_base + NQ <= nq_total) ? NQ : (nq_total - q_base);
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* sq = smem;
  float* sc = sq + static_cast<std::size_t>(NQ) * dim;
  float* qp = sc + n_cent;

  for (int qi = 0; qi < NQ; ++qi) {
    float* dst = sq + static_cast<std::size_t>(qi) * dim;
    if (qi < nq_local) {
      const float* src = q_scan + static_cast<std::size_t>(q_base + qi) * dim;
      for (int j = tid; j < dim; j += blockDim.x) dst[j] = src[j];
      if (tid == 0) qp[qi] = use_residual ? __ldg(q_perp + (q_base + qi)) : 0.f;
    } else {
      for (int j = tid; j < dim; j += blockDim.x) dst[j] = 0.f;
      if (tid == 0) qp[qi] = 0.f;
    }
  }
  for (int j = tid; j < n_cent; j += blockDim.x) sc[j] = centroids[j];
  __syncthreads();

  float cent[16];
#pragma unroll
  for (int c = 0; c < 16; ++c) cent[c] = (c < n_cent) ? sc[c] : 0.f;
  float qperp_reg[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) qperp_reg[qi] = qp[qi];

  float loc_s[NQ][kLocal];
  int loc_i[NQ][kLocal];
  int loc_sz[NQ];
  float thr_s[NQ];
#pragma unroll
  for (int qi = 0; qi < NQ; ++qi) {
    loc_sz[qi] = 0;
    thr_s[qi] = -1e30f;
  }
  const int kk = (k < kCap) ? k : kCap;

  const int n_tiles = (n + kGpuTile - 1) / kGpuTile;
  const int tile_stride = blockDim.x / kGpuTile;
  for (int tile = tid / kGpuTile; tile < n_tiles; tile += tile_stride) {
    const int lane = tid & (kGpuTile - 1);
    const int vi = tile * kGpuTile + lane;
    if (vi >= n) continue;
    const std::uint32_t* base =
        codes_u32 + static_cast<std::size_t>(tile) * nbg4 * kGpuTile + lane;
    const float scale = __ldg(scales + vi);
    const float resid = use_residual ? __ldg(residual_norms + vi) : 0.f;

    float scores[NQ];
    if constexpr (Bits == 2) {
      score_mq_u32_2bit<NQ>(base, sq, dim, cent, nbg4, scores);
    } else {
      score_mq_u32_4bit<NQ>(base, sq, dim, cent, nbg4, scores);
    }
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
      if (qi >= nq_local) continue;
      const float s = scores[qi] * scale + qperp_reg[qi] * resid;
      if (loc_sz[qi] >= kLocal && s <= thr_s[qi]) continue;
      heap_push_min(loc_s[qi], loc_i[qi], loc_sz[qi], kLocal, s, vi);
      thr_s[qi] = loc_s[qi][0];
    }
  }

  __shared__ float warp_s[kScanThreads / 32][kCap];
  __shared__ int warp_i[kScanThreads / 32][kCap];
  __shared__ int warp_n[kScanThreads / 32];
  const int warp = tid >> 5;
  const int lane = tid & 31;

  for (int qi = 0; qi < nq_local; ++qi) {
    if (lane == 0) warp_n[warp] = 0;
    __syncwarp();
    for (int t = 0; t < 32; ++t) {
      if (lane == t) {
        for (int i = 0; i < loc_sz[qi]; ++i) {
          heap_push_min(warp_s[warp], warp_i[warp], warp_n[warp], kk, loc_s[qi][i], loc_i[qi][i]);
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
      const int qg = q_base + qi;
      float* os = out_scores + static_cast<std::size_t>(qg) * k;
      unsigned long long* oi = out_ids + static_cast<std::size_t>(qg) * k;
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
    __syncthreads();
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
  free_p(b.d_residual_norms);
  free_p(b.d_centroids);
  free_p(b.d_boundaries);
  free_p(b.d_perm);
  free_p(b.d_signs);
  free_p(b.d_landmark_basis);
  free_p(b.d_queries);
  free_p(b.d_q_rot);
  free_p(b.d_q_land);
  free_p(b.d_q_perp);
  free_p(b.d_out_scores);
  free_p(b.d_out_ids);
  b.capacity_q = 0;
  b.capacity_k = 0;
}

void alloc_index_buffers(CudaBuffers& b, std::size_t n, std::size_t ambient_dim,
                         std::size_t scan_dim, std::size_t bits, std::size_t block,
                         float inv_sqrt_block, std::size_t n_landmarks) {
  free_buffers(b);
  b.n = n;
  b.dim = ambient_dim;
  b.scan_dim = scan_dim;
  b.n_landmarks = n_landmarks;
  b.bits = bits;
  b.n_cent = 1u << bits;
  b.n_byte_groups = scan_dim / (8 / bits);
  b.nbg4 = (b.n_byte_groups + kGroupsPerU32 - 1) / kGroupsPerU32;
  b.n_tiles = (n + kGpuTile - 1) / kGpuTile;
  b.block = block;
  b.inv_sqrt_block = inv_sqrt_block;
  b.use_residual = n_landmarks > 0;

  const std::size_t codes_u32 = b.n_tiles * b.nbg4 * kGpuTile;
  VC_CUDA(cudaMalloc(&b.d_vectors, n * scan_dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_codes, codes_u32 * sizeof(std::uint32_t)));
  VC_CUDA(cudaMalloc(&b.d_scales, n * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_centroids, b.n_cent * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_boundaries, (b.n_cent > 1 ? b.n_cent - 1 : 1) * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_perm, ambient_dim * sizeof(std::uint32_t)));
  VC_CUDA(cudaMalloc(&b.d_signs, ambient_dim * sizeof(float)));
  VC_CUDA(cudaMemset(b.d_codes, 0, codes_u32 * sizeof(std::uint32_t)));
  if (b.use_residual) {
    VC_CUDA(cudaMalloc(&b.d_residual_norms, n * sizeof(float)));
    VC_CUDA(cudaMalloc(&b.d_landmark_basis, ambient_dim * n_landmarks * sizeof(float)));
  }
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
  free_p(b.d_q_land);
  free_p(b.d_q_perp);
  free_p(b.d_out_scores);
  free_p(b.d_out_ids);

  b.capacity_q = std::max(nq, std::size_t{1});
  b.capacity_k = std::max(k, std::size_t{1});
  VC_CUDA(cudaMalloc(&b.d_queries, b.capacity_q * b.dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_q_rot, b.capacity_q * b.dim * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_out_scores, b.capacity_q * b.capacity_k * sizeof(float)));
  VC_CUDA(cudaMalloc(&b.d_out_ids, b.capacity_q * b.capacity_k * sizeof(std::uint64_t)));
  if (b.use_residual) {
    VC_CUDA(cudaMalloc(&b.d_q_land, b.capacity_q * b.scan_dim * sizeof(float)));
    VC_CUDA(cudaMalloc(&b.d_q_perp, b.capacity_q * sizeof(float)));
  }
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
  VC_CUDA(cudaMemcpy(b.d_vectors, host_vectors, n * b.scan_dim * sizeof(float),
                     cudaMemcpyHostToDevice));
}

void upload_residuals(CudaBuffers& b, const float* host_residuals, std::size_t n) {
  if (!b.d_residual_norms) throw std::runtime_error("upload_residuals without residual buffer");
  VC_CUDA(cudaMemcpy(b.d_residual_norms, host_residuals, n * sizeof(float), cudaMemcpyHostToDevice));
}

void upload_landmark_basis(CudaBuffers& b, const float* host_L, std::size_t dim, std::size_t m) {
  if (!b.d_landmark_basis) throw std::runtime_error("upload_landmark_basis without basis buffer");
  VC_CUDA(cudaMemcpy(b.d_landmark_basis, host_L, dim * m * sizeof(float), cudaMemcpyHostToDevice));
}

void encode_on_device(CudaBuffers& b) {
  encode_coop<<<static_cast<int>(b.n), 256, b.scan_dim * sizeof(float)>>>(
      b.d_vectors, b.d_codes, b.d_scales, b.d_centroids, b.d_boundaries, b.d_perm, b.d_signs,
      static_cast<int>(b.n), static_cast<int>(b.scan_dim), static_cast<int>(b.bits),
      static_cast<int>(b.n_cent), static_cast<int>(b.n_cent > 1 ? b.n_cent - 1 : 0),
      static_cast<int>(b.n_byte_groups), static_cast<int>(b.nbg4), static_cast<int>(b.block),
      b.inv_sqrt_block);
  VC_CUDA(cudaGetLastError());
  cuda_sync();
}

void encode_prerotated_on_device(CudaBuffers& b) {
  encode_prerotated_coop<<<static_cast<int>(b.n), 256, b.scan_dim * sizeof(float)>>>(
      b.d_vectors, b.d_codes, b.d_scales, b.d_centroids, b.d_boundaries, static_cast<int>(b.n),
      static_cast<int>(b.scan_dim), static_cast<int>(b.bits), static_cast<int>(b.n_cent),
      static_cast<int>(b.n_cent > 1 ? b.n_cent - 1 : 0), static_cast<int>(b.n_byte_groups),
      static_cast<int>(b.nbg4));
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

  const float* q_scan = b.d_q_rot;
  if (b.use_residual) {
    project_queries_landmarks<<<static_cast<int>(nq), 256, b.dim * sizeof(float)>>>(
        b.d_q_rot, b.d_landmark_basis, b.d_q_land, b.d_q_perp, static_cast<int>(nq),
        static_cast<int>(b.dim), static_cast<int>(b.n_landmarks), static_cast<int>(b.scan_dim));
    VC_CUDA(cudaGetLastError());
    q_scan = b.d_q_land;
  }

  {
    const int ni = static_cast<int>(b.n);
    const int nqi = static_cast<int>(nq);
    const int di = static_cast<int>(b.scan_dim);
    const int nc = static_cast<int>(b.n_cent);
    const int nbg4 = static_cast<int>(b.nbg4);
    const int ki = static_cast<int>(k);
    const int bits = static_cast<int>(b.bits);
    const int use_res = b.use_residual ? 1 : 0;

    auto launch_mq = [&](auto nq_tag, auto k_tag, auto local_tag, auto bits_tag) {
      constexpr int NQ = decltype(nq_tag)::value;
      constexpr int kCap = decltype(k_tag)::value;
      constexpr int kLocal = decltype(local_tag)::value;
      constexpr int Bits = decltype(bits_tag)::value;
      const std::size_t shmem =
          (static_cast<std::size_t>(NQ) * b.scan_dim + b.n_cent + NQ) * sizeof(float);
      const int grid = static_cast<int>((nq + NQ - 1) / NQ);
      scan_mq_topk<NQ, kCap, kLocal, Bits><<<grid, kScanThreads, shmem>>>(
          b.d_codes, b.d_scales, b.d_residual_norms, q_scan, b.d_q_perp, b.d_centroids,
          b.d_out_scores, b.d_out_ids, ni, nqi, di, nc, nbg4, ki, use_res);
    };

    auto launch_bits = [&](auto nq_tag, auto k_tag, auto local_tag) {
      if (bits == 2) {
        launch_mq(nq_tag, k_tag, local_tag, std::integral_constant<int, 2>{});
      } else {
        launch_mq(nq_tag, k_tag, local_tag, std::integral_constant<int, 4>{});
      }
    };

    if (k <= 16) {
      if (nq >= 4) {
        launch_bits(std::integral_constant<int, kNqBatch>{}, std::integral_constant<int, 16>{},
                    std::integral_constant<int, 4>{});
      } else {
        launch_bits(std::integral_constant<int, 1>{}, std::integral_constant<int, 16>{},
                    std::integral_constant<int, 4>{});
      }
    } else if (k <= 64) {
      if (nq >= 4) {
        launch_bits(std::integral_constant<int, kNqBatch>{}, std::integral_constant<int, 64>{},
                    std::integral_constant<int, 8>{});
      } else {
        launch_bits(std::integral_constant<int, 1>{}, std::integral_constant<int, 64>{},
                    std::integral_constant<int, 8>{});
      }
    } else {
      launch_bits(std::integral_constant<int, kNqBatchLargeK>{},
                  std::integral_constant<int, 256>{}, std::integral_constant<int, 16>{});
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
