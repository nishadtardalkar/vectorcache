#include "vectorcache/search/search.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/encode/encode.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/landmark_scan.hpp"
#include "vectorcache/search/search_avx2.hpp"
#include "vectorcache/search/search_vnni.hpp"
#include "vectorcache/search/topk_heap.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

constexpr std::size_t kNqBatch = 8;
/// turbovec ST permute-dot widen: fewer passes when a single thread owns the scan.
constexpr std::size_t kNqBatchWide = 10;
constexpr std::size_t kTilesPerThread = 32;
constexpr std::size_t kMinTileBlocks = 1024;
constexpr std::size_t kMinTileBlocksX86 = kMinTileBlocks * 3;
constexpr std::size_t kSingleQueryParallelMinBlocks = 1024;

#if defined(__x86_64__) || defined(_M_X64)
inline float hsum256_ps(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s = _mm_add_ps(lo, hi);
  s = _mm_add_ps(s, _mm_movehdup_ps(s));
  s = _mm_add_ss(s, _mm_movehl_ps(s, s));
  return _mm_cvtss_f32(s);
}

/// q_land[qi, 0:m) = landmark_basis[0:m, :] · q_rot[qi, :]
/// Landmark-tile outer keeps L hot; query micro-tile reuses each L load.
void project_onto_landmarks(const float* q_rot, std::size_t nq, std::size_t dim,
                            const float* landmark_basis, std::size_t n_landmarks, float* q_land,
                            std::size_t scan_dim) {
  constexpr std::size_t kLmTile = 8;
  constexpr std::size_t kQTile = 4;
  const std::size_t m_aligned = n_landmarks & ~(kLmTile - 1);
  const std::size_t nq_aligned = nq - (nq % kQTile);

  for (std::size_t j0 = 0; j0 < m_aligned; j0 += kLmTile) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi0 = 0; qi0 < static_cast<int>(nq_aligned); qi0 += static_cast<int>(kQTile)) {
      const std::size_t base = static_cast<std::size_t>(qi0);
      __m256 acc[kQTile][kLmTile];
      for (std::size_t r = 0; r < kQTile; ++r)
        for (std::size_t t = 0; t < kLmTile; ++t) acc[r][t] = _mm256_setzero_ps();
      for (std::size_t i = 0; i < dim; i += 8) {
        __m256 Lv[kLmTile];
        for (std::size_t t = 0; t < kLmTile; ++t) {
          Lv[t] = _mm256_loadu_ps(landmark_basis + (j0 + t) * dim + i);
        }
        for (std::size_t r = 0; r < kQTile; ++r) {
          const __m256 qv = _mm256_loadu_ps(q_rot + (base + r) * dim + i);
          for (std::size_t t = 0; t < kLmTile; ++t) {
            acc[r][t] = _mm256_fmadd_ps(qv, Lv[t], acc[r][t]);
          }
        }
      }
      for (std::size_t r = 0; r < kQTile; ++r) {
        float* dest = q_land + (base + r) * scan_dim + j0;
        for (std::size_t t = 0; t < kLmTile; ++t) dest[t] = hsum256_ps(acc[r][t]);
      }
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = static_cast<int>(nq_aligned); qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      __m256 acc[kLmTile];
      for (std::size_t t = 0; t < kLmTile; ++t) acc[t] = _mm256_setzero_ps();
      for (std::size_t i = 0; i < dim; i += 8) {
        const __m256 qv = _mm256_loadu_ps(q_rot + qis * dim + i);
        for (std::size_t t = 0; t < kLmTile; ++t) {
          acc[t] = _mm256_fmadd_ps(
              qv, _mm256_loadu_ps(landmark_basis + (j0 + t) * dim + i), acc[t]);
        }
      }
      float* dest = q_land + qis * scan_dim + j0;
      for (std::size_t t = 0; t < kLmTile; ++t) dest[t] = hsum256_ps(acc[t]);
    }
  }

  for (std::size_t j = m_aligned; j < n_landmarks; ++j) {
    const float* Lj = landmark_basis + j * dim;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const float* q = q_rot + static_cast<std::size_t>(qi) * dim;
      __m256 acc = _mm256_setzero_ps();
      for (std::size_t i = 0; i < dim; i += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(q + i), _mm256_loadu_ps(Lj + i), acc);
      }
      q_land[static_cast<std::size_t>(qi) * scan_dim + j] = hsum256_ps(acc);
    }
  }
}
#else
void project_onto_landmarks(const float* q_rot, std::size_t nq, std::size_t dim,
                            const float* landmark_basis, std::size_t n_landmarks, float* q_land,
                            std::size_t scan_dim) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    const float* row = q_rot + static_cast<std::size_t>(qi) * dim;
    float* dest = q_land + static_cast<std::size_t>(qi) * scan_dim;
    for (std::size_t j = 0; j < n_landmarks; ++j) {
      float s = 0.f;
      const float* Lj = landmark_basis + j * dim;
      for (std::size_t i = 0; i < dim; ++i) s += row[i] * Lj[i];
      dest[j] = s;
    }
  }
}
#endif

std::size_t nq_batch_for(bool use_permute_dot, std::size_t n_threads, std::size_t nq) {
  if (use_permute_dot && n_threads <= 1 && (nq + kNqBatchWide - 1) / kNqBatchWide <
                                               (nq + kNqBatch - 1) / kNqBatch) {
    return kNqBatchWide;
  }
  return kNqBatch;
}

PreparedQueryLut build_query_lut(std::span<const float> q_rot_row, std::span<const float> centroids,
                                 std::size_t bits, std::size_t dim) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t codes_per_nibble = codes_per_byte / 2;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  const std::uint16_t code_mask = static_cast<std::uint16_t>((1u << bits) - 1);

  PreparedQueryLut out;
  out.uint8_luts.assign(n_byte_groups * 32, 0);
  std::vector<float> float_vals(n_byte_groups * 32);
  std::vector<float> mins(n_byte_groups * 2);
  float max_span = 0.f;
  out.bias = 0.f;

  for (std::size_t g = 0; g < n_byte_groups; ++g) {
    const std::size_t dim_start = g * codes_per_byte;
    float lo_min = std::numeric_limits<float>::max();
    float lo_max = -std::numeric_limits<float>::max();
    for (std::uint16_t nibble_val = 0; nibble_val < 16; ++nibble_val) {
      float s = 0.f;
      for (std::size_t c = 0; c < codes_per_nibble; ++c) {
        const auto shift = (codes_per_nibble - 1 - c) * bits;
        const auto code = (nibble_val >> shift) & code_mask;
        s += q_rot_row[dim_start + c] * centroids[code];
      }
      float_vals[g * 32 + nibble_val] = s;
      lo_min = std::min(lo_min, s);
      lo_max = std::max(lo_max, s);
    }
    float hi_min = std::numeric_limits<float>::max();
    float hi_max = -std::numeric_limits<float>::max();
    for (std::uint16_t nibble_val = 0; nibble_val < 16; ++nibble_val) {
      float s = 0.f;
      for (std::size_t c = 0; c < codes_per_nibble; ++c) {
        const auto shift = (codes_per_nibble - 1 - c) * bits;
        const auto code = (nibble_val >> shift) & code_mask;
        s += q_rot_row[dim_start + codes_per_nibble + c] * centroids[code];
      }
      float_vals[g * 32 + 16 + nibble_val] = s;
      hi_min = std::min(hi_min, s);
      hi_max = std::max(hi_max, s);
    }
    mins[g * 2] = lo_min;
    mins[g * 2 + 1] = hi_min;
    out.bias += lo_min + hi_min;
    max_span = std::max(max_span, lo_max - lo_min);
    max_span = std::max(max_span, hi_max - hi_min);
  }

  constexpr float kMaxLut = 127.f;
  out.scale = (max_span > 0.f) ? (max_span / kMaxLut) : 1.f;
  float inv_scale = 1.f;
  if (out.scale >= std::numeric_limits<float>::min()) {
    inv_scale = 1.f / out.scale;
  } else {
    out.scale = 1.f;
  }
  for (std::size_t g = 0; g < n_byte_groups; ++g) {
    const float lo_min = mins[g * 2];
    const float hi_min = mins[g * 2 + 1];
    for (int i = 0; i < 16; ++i) {
      const std::size_t j_lo = g * 32 + static_cast<std::size_t>(i);
      const std::size_t j_hi = g * 32 + 16 + static_cast<std::size_t>(i);
      out.uint8_luts[j_lo] = static_cast<std::uint8_t>(
          std::clamp(std::round((float_vals[j_lo] - lo_min) * inv_scale), 0.f, kMaxLut));
      out.uint8_luts[j_hi] = static_cast<std::uint8_t>(
          std::clamp(std::round((float_vals[j_hi] - hi_min) * inv_scale), 0.f, kMaxLut));
    }
  }
  return out;
}

void score_query_scalar(const PreparedQueryLut& lut, std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t bits,
                        std::size_t n_byte_groups, std::size_t n_vectors, std::size_t n_blocks,
                        std::size_t k, float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                        float& heap_min, std::size_t& heap_mi, float bias_corr,
                        const std::uint64_t* id_map, float score_offset) {
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::size_t vi = base_vec + lane;
      if (vi >= n_vectors) break;
      float score = lut.bias + bias_corr;
      for (std::size_t g = 0; g < n_byte_groups; ++g) {
        const auto byte_val =
            static_cast<std::size_t>(read_code(blocked_codes, bits, n_byte_groups, b, g, lane));
        score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + (byte_val >> 4)]);
        score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + 16 + (byte_val & 0x0F)]);
      }
      score = score * vec_scales[vi] + score_offset;
      heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k, score,
                           topk_map_id(id_map, vi));
    }
  }
}

#if defined(__x86_64__) || defined(_M_X64)
bool cpu_has_avx2() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2");
#elif defined(_MSC_VER)
  int info[4];
  __cpuid(info, 0);
  if (info[0] >= 7) {
    __cpuidex(info, 7, 0);
    return (info[1] & (1 << 5)) != 0;
  }
  return false;
#else
  return false;
#endif
}
#endif

bool single_query_parallelizes(std::size_t n_vectors) {
  const std::size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
  return n_blocks >= kSingleQueryParallelMinBlocks;
}

std::size_t range_cap_for_k(std::size_t n_vectors, std::size_t k) {
  constexpr std::size_t kMinVectorsPerRangePerK = 512;
  return std::max<std::size_t>(1, (n_vectors + kMinVectorsPerRangePerK * std::max(k, std::size_t{1}) -
                                   1) /
                                      (kMinVectorsPerRangePerK * std::max(k, std::size_t{1})));
}

std::size_t n_block_ranges(std::size_t nq, std::size_t n_quads, std::size_t n_blocks,
                           std::size_t n_vectors, std::size_t k, std::size_t n_threads,
                           std::size_t tiles_per_thread, std::size_t min_tile_blocks, bool serial) {
  if (n_threads <= 1 || serial || (nq == 1 && !single_query_parallelizes(n_vectors))) {
    return 1;
  }
  const std::size_t by_threads = (n_threads * tiles_per_thread + n_quads - 1) / n_quads;
  const std::size_t by_blocks = (n_blocks + min_tile_blocks - 1) / min_tile_blocks;
  return std::max<std::size_t>(
      1, std::min({by_threads, by_blocks, range_cap_for_k(n_vectors, k)}));
}

std::size_t smooth_tile_count(std::size_t n_ranges, std::size_t n_quads, std::size_t n_threads) {
  const std::size_t tiles = n_quads * n_ranges;
  if (tiles > n_threads && tiles < 2 * n_threads) {
    return std::max<std::size_t>(1, n_threads / n_quads);
  }
  return n_ranges;
}

/// turbovec `block_range_stride`: ~one large range per thread for nq=1 MT.
std::size_t block_range_stride(std::size_t n_blocks, std::size_t n_threads) {
  std::size_t s = (n_blocks + n_threads - 1) / std::max(n_threads, std::size_t{1});
  if (s < 64) s = 64;
  if ((s & 1u) != 0) ++s;
  return s;
}

struct TileCand {
  float score;
  std::uint64_t id;
};

void write_sorted_topk(SearchResults& out, std::size_t qi, std::size_t effective_k,
                       std::vector<float>& heap_s, std::vector<std::uint64_t>& heap_i,
                       std::size_t heap_sz) {
  std::vector<std::size_t> order(heap_sz);
  for (std::size_t i = 0; i < heap_sz; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (heap_s[a] != heap_s[b]) return heap_s[a] > heap_s[b];
    return heap_i[a] < heap_i[b];
  });
  for (std::size_t i = 0; i < heap_sz; ++i) {
    out.scores[qi * effective_k + i] = heap_s[order[i]];
    out.ids[qi * effective_k + i] = heap_i[order[i]];
  }
}

void remap_ids(SearchResults& out, std::span<const std::uint64_t> id_map) {
  if (id_map.empty()) return;
  for (std::uint64_t& id : out.ids) {
    if (id >= id_map.size()) {
      throw std::out_of_range("score_prepared: local id out of id_map range");
    }
    id = id_map[static_cast<std::size_t>(id)];
  }
}

SearchResults score_prepared_vm(const PreparedQueries& prep, std::size_t effective_k,
                                std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                                std::span<const float> scales,
                                std::span<const std::size_t> query_indices,
                                const float* residual_norms = nullptr,
                                const float* residual_scales = nullptr,
                                LandmarkStageTimings* timings = nullptr,
                                const float* scale_resid = nullptr) {
  (void)timings;
  const std::size_t nq = query_indices.empty() ? prep.nq : query_indices.size();
  auto global_qi = [&](std::size_t local_i) -> std::size_t {
    return query_indices.empty() ? local_i : query_indices[local_i];
  };
  const std::size_t n_vectors = scales.size();
  const std::size_t n_byte_groups = prep.n_byte_groups;
  const bool use_pd = prep.backend == SearchBackendKind::VmPermuteDot;

  SearchResults out;
  out.nq = nq;
  out.k = effective_k;
  out.scores.assign(nq * effective_k, 0.f);
  out.ids.assign(nq * effective_k, 0);
  if (nq == 0) {
    return out;
  }

  std::size_t n_threads = 1;
#ifdef _OPENMP
  n_threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif

  // turbovec: VM units align to block boundaries (block_bytes % 128 == 0 when
  // n_byte_groups % 4 == 0), so block-range subspans are valid — same as their
  // `blocked_codes[block_start * block_bytes ..]`.
  // nq=1 uses turbovec `block_range_stride`; multi-query uses TILES_PER_THREAD tiling.
  const std::size_t nq_batch = nq_batch_for(use_pd, n_threads, nq);
  const std::size_t n_quads = (nq + nq_batch - 1) / nq_batch;
  std::size_t blocks_per_range = 0;
  std::size_t n_ranges = 1;
  if (nq == 1 && n_threads > 1 && single_query_parallelizes(n_vectors)) {
    blocks_per_range = block_range_stride(n_blocks, n_threads);
    n_ranges = (n_blocks + blocks_per_range - 1) / blocks_per_range;
  } else {
    n_ranges = n_block_ranges(nq, n_quads, n_blocks, n_vectors, effective_k, n_threads,
                              kTilesPerThread, kMinTileBlocksX86, false);
    n_ranges = smooth_tile_count(n_ranges, n_quads, n_threads);
    blocks_per_range = (n_blocks + n_ranges - 1) / n_ranges;
  }

  struct Tile {
    std::size_t qi_start;
    std::size_t block_start;
  };
  std::vector<Tile> tiles;
  tiles.reserve(n_quads * n_ranges);
  for (std::size_t block_start = 0; block_start < n_blocks; block_start += blocks_per_range) {
    for (std::size_t qi_start = 0; qi_start < nq; qi_start += nq_batch) {
      tiles.push_back(Tile{qi_start, block_start});
    }
  }

  std::vector<std::vector<TileCand>> all_cands(nq);
  // turbovec seed_cell: publish filled-heap mins so later tiles prune harder.
  std::vector<std::atomic<std::uint32_t>> prune_seed(nq);
  {
    const float ninf = -std::numeric_limits<float>::infinity();
    std::uint32_t ninf_bits = 0;
    std::memcpy(&ninf_bits, &ninf, sizeof(ninf_bits));
    for (std::size_t i = 0; i < nq; ++i) {
      prune_seed[i].store(ninf_bits, std::memory_order_relaxed);
    }
  }
  // MinGW Win64 miscompiles concurrent AVX-512 VNNI under OpenMP; keep the VM
  // tile loop serial there. MSVC / Linux keep query×block tiling.
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel
#endif
  {
    std::vector<std::vector<TileCand>> local_cands(nq);
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp for schedule(dynamic)
#endif
    for (int ti = 0; ti < static_cast<int>(tiles.size()); ++ti) {
      const Tile tile = tiles[static_cast<std::size_t>(ti)];
      const std::size_t qi_end = std::min(tile.qi_start + nq_batch, nq);
      const std::size_t batch_nq = qi_end - tile.qi_start;
      const std::size_t block_end = std::min(tile.block_start + blocks_per_range, n_blocks);
      const std::size_t range_blocks = block_end - tile.block_start;
      if (range_blocks == 0 || batch_nq == 0) continue;

      const std::size_t vec_start = tile.block_start * kBlock;
      const std::size_t vec_end = std::min(block_end * kBlock, n_vectors);
      const std::size_t range_n = vec_end - vec_start;
      const std::size_t block_bytes = n_byte_groups * kBlock;
      const std::size_t code_off = tile.block_start * block_bytes;
      const std::size_t code_len = range_blocks * block_bytes;

      auto codes_span = blocked_codes.subspan(code_off, code_len);
      auto scales_span = scales.subspan(vec_start, range_n);

      std::vector<float> heap_s_store(batch_nq * effective_k);
      std::vector<std::uint64_t> heap_i_store(batch_nq * effective_k);
      std::vector<std::size_t> heap_sz(batch_nq, 0);
      std::vector<float> heap_min(batch_nq);
      std::vector<std::size_t> heap_mi(batch_nq, 0);
      std::vector<float*> heap_s_ptrs(batch_nq);
      std::vector<std::uint64_t*> heap_i_ptrs(batch_nq);
      for (std::size_t i = 0; i < batch_nq; ++i) {
        heap_s_ptrs[i] = heap_s_store.data() + i * effective_k;
        heap_i_ptrs[i] = heap_i_store.data() + i * effective_k;
        const std::uint32_t seed_bits =
            prune_seed[tile.qi_start + i].load(std::memory_order_acquire);
        std::memcpy(&heap_min[i], &seed_bits, sizeof(float));
      }

      const float* range_resid =
          residual_norms != nullptr ? residual_norms + vec_start : nullptr;
      const float* range_sr = scale_resid != nullptr ? scale_resid + vec_start * 2 : nullptr;
      std::vector<float> rs_batch;
      const float* rs_ptr = nullptr;
      if (residual_scales != nullptr) {
        rs_batch.resize(batch_nq);
        for (std::size_t i = 0; i < batch_nq; ++i) {
          rs_batch[i] = residual_scales[global_qi(tile.qi_start + i)];
        }
        rs_ptr = rs_batch.data();
      }
      if (use_pd) {
        std::vector<const QueryPermuteDot*> pd_ptrs(batch_nq);
        for (std::size_t i = 0; i < batch_nq; ++i) {
          pd_ptrs[i] = &prep.pds[global_qi(tile.qi_start + i)];
        }
        score_queries_permute_dot(pd_ptrs.data(), batch_nq, codes_span, scales_span, n_byte_groups,
                                  range_n, range_blocks, effective_k, heap_s_ptrs.data(),
                                  heap_i_ptrs.data(), heap_sz.data(), heap_min.data(),
                                  heap_mi.data(), nullptr, nullptr, nullptr, 0, 0, range_resid,
                                  rs_ptr, range_sr);
      } else {
        std::vector<const std::uint8_t*> lut_ptrs(batch_nq);
        std::vector<float> scales_batch(batch_nq);
        std::vector<float> biases_batch(batch_nq);
        for (std::size_t i = 0; i < batch_nq; ++i) {
          const std::size_t gqi = global_qi(tile.qi_start + i);
          lut_ptrs[i] = prep.split_luts[gqi].data();
          scales_batch[i] = prep.lut_scales[gqi];
          biases_batch[i] = prep.lut_biases[gqi];
        }
        score_queries_vnni(lut_ptrs.data(), scales_batch.data(), biases_batch.data(), batch_nq,
                           codes_span, scales_span, n_byte_groups, range_n, range_blocks, effective_k,
                           heap_s_ptrs.data(), heap_i_ptrs.data(), heap_sz.data(), heap_min.data(),
                           heap_mi.data(), nullptr, nullptr, range_resid, rs_ptr);
      }
      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t qi = tile.qi_start + i;
        if (heap_sz[i] >= effective_k) {
          // Atomic max of heap_min bits (valid prune seed for later tiles).
          std::uint32_t want = 0;
          std::memcpy(&want, &heap_min[i], sizeof(want));
          std::uint32_t prev = prune_seed[qi].load(std::memory_order_relaxed);
          while (true) {
            float prev_f = 0.f;
            float want_f = 0.f;
            std::memcpy(&prev_f, &prev, sizeof(prev_f));
            std::memcpy(&want_f, &want, sizeof(want_f));
            if (!(want_f > prev_f)) break;
            if (prune_seed[qi].compare_exchange_weak(prev, want, std::memory_order_release,
                                                     std::memory_order_relaxed)) {
              break;
            }
          }
        }
        auto& dest = local_cands[qi];
        dest.reserve(dest.size() + heap_sz[i]);
        for (std::size_t j = 0; j < heap_sz[i]; ++j) {
          dest.push_back(TileCand{heap_s_ptrs[i][j],
                                  heap_i_ptrs[i][j] + static_cast<std::uint64_t>(vec_start)});
        }
      }
    }

#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp critical
#endif
    {
      for (std::size_t qi = 0; qi < nq; ++qi) {
        if (!local_cands[qi].empty()) {
          all_cands[qi].insert(all_cands[qi].end(), local_cands[qi].begin(),
                               local_cands[qi].end());
        }
      }
    }
  }

  for (std::size_t qi = 0; qi < nq; ++qi) {
    auto& cands = all_cands[qi];
    const std::size_t kk = std::min(effective_k, cands.size());
    std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(kk), cands.end(),
                      [](const TileCand& a, const TileCand& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.id < b.id;
                      });
    for (std::size_t j = 0; j < kk; ++j) {
      out.scores[qi * effective_k + j] = cands[j].score;
      out.ids[qi * effective_k + j] = cands[j].id;
    }
  }
  return out;
}

SearchResults score_prepared_perm0(const PreparedQueries& prep, std::size_t effective_k,
                                   std::span<const std::uint8_t> blocked_codes,
                                   std::size_t n_blocks, std::span<const float> scales,
                                   std::span<const std::size_t> query_indices) {
  const std::size_t nq = query_indices.empty() ? prep.nq : query_indices.size();
  auto global_qi = [&](std::size_t local_i) -> std::size_t {
    return query_indices.empty() ? local_i : query_indices[local_i];
  };
  const std::size_t n_vectors = scales.size();
  const bool use_avx2 = prep.backend == SearchBackendKind::Avx2;
  const std::size_t n_byte_groups = prep.n_byte_groups;

  SearchResults out;
  out.nq = nq;
  out.k = effective_k;
  out.scores.assign(nq * effective_k, 0.f);
  out.ids.assign(nq * effective_k, 0);
  if (nq == 0) return out;

  // Scalar / non-x86: one query at a time.
  if (!use_avx2) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t gqi = global_qi(static_cast<std::size_t>(qi));
      const auto& lut = prep.luts[gqi];
      std::vector<float> heap_s(effective_k);
      std::vector<std::uint64_t> heap_i(effective_k);
      std::size_t heap_sz = 0;
      float heap_min = -std::numeric_limits<float>::infinity();
      std::size_t heap_mi = 0;
      score_query_scalar(lut, blocked_codes, scales, prep.bits, n_byte_groups, n_vectors, n_blocks,
                         effective_k, heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi,
                         prep.bias_corrs[gqi], nullptr, 0.f);
      write_sorted_topk(out, static_cast<std::size_t>(qi), effective_k, heap_s, heap_i, heap_sz);
    }
    return out;
  }

#if defined(__x86_64__) || defined(_M_X64)
  // turbovec-style: tile (query-batch × block-range); fused NQ<=4 shares code loads.
  std::size_t n_threads = 1;
#ifdef _OPENMP
  n_threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
  const std::size_t n_quads = (nq + kAvx2NqBatch - 1) / kAvx2NqBatch;
  std::size_t n_ranges =
      n_block_ranges(nq, n_quads, n_blocks, n_vectors, effective_k, n_threads, kTilesPerThread,
                     kMinTileBlocksX86, false);
  n_ranges = smooth_tile_count(n_ranges, n_quads, n_threads);
  const std::size_t blocks_per_range = (n_blocks + n_ranges - 1) / n_ranges;

  struct Tile {
    std::size_t qi_start;
    std::size_t block_start;
  };
  std::vector<Tile> tiles;
  tiles.reserve(n_quads * n_ranges);
  for (std::size_t block_start = 0; block_start < n_blocks; block_start += blocks_per_range) {
    for (std::size_t qi_start = 0; qi_start < nq; qi_start += kAvx2NqBatch) {
      tiles.push_back(Tile{qi_start, block_start});
    }
  }

  std::vector<std::vector<TileCand>> all_cands(nq);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<std::vector<TileCand>> local_cands(nq);
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (int ti = 0; ti < static_cast<int>(tiles.size()); ++ti) {
      const Tile tile = tiles[static_cast<std::size_t>(ti)];
      const std::size_t qi_end = std::min(tile.qi_start + kAvx2NqBatch, nq);
      const std::size_t batch_nq = qi_end - tile.qi_start;
      const std::size_t block_end = std::min(tile.block_start + blocks_per_range, n_blocks);
      const std::size_t range_blocks = block_end - tile.block_start;
      if (range_blocks == 0 || batch_nq == 0) continue;

      const std::size_t vec_start = tile.block_start * kBlock;
      const std::size_t vec_end = std::min(block_end * kBlock, n_vectors);
      const std::size_t range_n = vec_end - vec_start;
      const std::size_t block_bytes = n_byte_groups * kBlock;
      const std::size_t code_off = tile.block_start * block_bytes;
      const std::size_t code_len = range_blocks * block_bytes;

      auto codes_span = blocked_codes.subspan(code_off, code_len);
      auto scales_span = scales.subspan(vec_start, range_n);

      std::vector<float> heap_s_store(batch_nq * effective_k);
      std::vector<std::uint64_t> heap_i_store(batch_nq * effective_k);
      std::vector<std::size_t> heap_sz(batch_nq, 0);
      std::vector<float> heap_min(batch_nq, -std::numeric_limits<float>::infinity());
      std::vector<std::size_t> heap_mi(batch_nq, 0);
      float* heap_s_ptrs[kAvx2NqBatch];
      std::uint64_t* heap_i_ptrs[kAvx2NqBatch];
      QueryLutView lut_views[kAvx2NqBatch];
      float bias_corrs[kAvx2NqBatch];
      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t gqi = global_qi(tile.qi_start + i);
        heap_s_ptrs[i] = heap_s_store.data() + i * effective_k;
        heap_i_ptrs[i] = heap_i_store.data() + i * effective_k;
        const auto& lut = prep.luts[gqi];
        lut_views[i] = QueryLutView{lut.uint8_luts.data(), lut.scale, lut.bias};
        bias_corrs[i] = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      }

      score_queries_avx2_perm0(lut_views, batch_nq, codes_span, scales_span, n_byte_groups, range_n,
                               range_blocks, effective_k, heap_s_ptrs, heap_i_ptrs, heap_sz.data(),
                               heap_min.data(), heap_mi.data(), bias_corrs, nullptr, nullptr);

      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t qi = tile.qi_start + i;
        auto& dest = local_cands[qi];
        dest.reserve(dest.size() + heap_sz[i]);
        for (std::size_t j = 0; j < heap_sz[i]; ++j) {
          dest.push_back(TileCand{heap_s_ptrs[i][j],
                                  heap_i_ptrs[i][j] + static_cast<std::uint64_t>(vec_start)});
        }
      }
    }

#ifdef _OPENMP
#pragma omp critical
#endif
    {
      for (std::size_t qi = 0; qi < nq; ++qi) {
        if (!local_cands[qi].empty()) {
          all_cands[qi].insert(all_cands[qi].end(), local_cands[qi].begin(),
                               local_cands[qi].end());
        }
      }
    }
  }

  for (std::size_t qi = 0; qi < nq; ++qi) {
    auto& cands = all_cands[qi];
    const std::size_t kk = std::min(effective_k, cands.size());
    if (kk == 0) continue;
    std::partial_sort(cands.begin(), cands.begin() + static_cast<std::ptrdiff_t>(kk), cands.end(),
                      [](const TileCand& a, const TileCand& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.id < b.id;
                      });
    for (std::size_t j = 0; j < kk; ++j) {
      out.scores[qi * effective_k + j] = cands[j].score;
      out.ids[qi * effective_k + j] = cands[j].id;
    }
  }
#else
  (void)n_byte_groups;
#endif

  return out;
}

}  // namespace

PreparedQueries prepare_queries(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                const Rotation& rotation, std::span<const float> centroids,
                                std::size_t bits, std::span<const float> tqplus_shift,
                                std::span<const float> tqplus_scale) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;

  PreparedQueries prep;
  prep.nq = nq;
  prep.dim = dim;
  prep.bits = bits;
  prep.n_byte_groups = n_byte_groups;
  prep.q_rot.assign(nq * dim, 0.f);
  prep.bias_corrs.assign(nq, 0.f);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> scratch(dim);
    std::vector<float> row(dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      auto q = queries.subspan(static_cast<std::size_t>(qi) * dim, dim);
      const float nrm = row_norm(q);
      const float inv = (nrm > kMinInputNorm) ? (1.f / nrm) : 0.f;
      rotation.apply_scaled_into(q, inv, row, scratch);
      auto dest = std::span<float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
      if (!tqplus_shift.empty()) {
        double bc = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
          dest[d] = row[d] / tqplus_scale[d];
          bc -= static_cast<double>(row[d]) * static_cast<double>(tqplus_shift[d]);
        }
        prep.bias_corrs[static_cast<std::size_t>(qi)] = static_cast<float>(bc);
      } else {
        std::copy(row.begin(), row.end(), dest.begin());
      }
    }
  }

  const bool use_vm = vector_major_for(bits, n_byte_groups);
#if defined(__x86_64__) || defined(_M_X64)
  if (use_vm) {
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
    const bool use_pd = false;
#else
    const bool use_pd = (bits == 4);
#endif
    prep.backend = use_pd ? SearchBackendKind::VmPermuteDot : SearchBackendKind::VmVnni;
    if (use_pd) {
      prep.pds.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        auto qrow =
            std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
        prep.pds[static_cast<std::size_t>(qi)] = build_permute_dot(qrow, centroids, dim);
        prep.pds[static_cast<std::size_t>(qi)].bias += prep.bias_corrs[static_cast<std::size_t>(qi)];
      }
    } else {
      prep.split_luts.resize(nq);
      prep.lut_scales.resize(nq);
      prep.lut_biases.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        auto qrow =
            std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
        PreparedQueryLut lut = build_query_lut(qrow, centroids, bits, dim);
        prep.split_luts[static_cast<std::size_t>(qi)] =
            split_lut_for_vnni(lut.uint8_luts, n_byte_groups);
        prep.lut_scales[static_cast<std::size_t>(qi)] = lut.scale;
        prep.lut_biases[static_cast<std::size_t>(qi)] =
            lut.bias + prep.bias_corrs[static_cast<std::size_t>(qi)];
      }
    }
    return prep;
  }
  prep.backend = cpu_has_avx2() ? SearchBackendKind::Avx2 : SearchBackendKind::Scalar;
#else
  (void)use_vm;
  prep.backend = SearchBackendKind::Scalar;
#endif

  prep.luts.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    auto qrow = std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
    prep.luts[static_cast<std::size_t>(qi)] = build_query_lut(qrow, centroids, bits, dim);
  }
  return prep;
}

PreparedQueries prepare_from_rows(std::span<const float> rows, std::size_t nq, std::size_t dim,
                                  std::span<const float> centroids, std::size_t bits) {
  if (rows.size() < nq * dim) {
    throw std::invalid_argument("prepare_from_rows: rows too short");
  }
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;

  PreparedQueries prep;
  prep.nq = nq;
  prep.dim = dim;
  prep.bits = bits;
  prep.n_byte_groups = n_byte_groups;
  prep.q_rot.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(nq * dim));
  prep.bias_corrs.assign(nq, 0.f);

  // Same backend selection as prepare_queries (vector-major permute_dot / VNNI when available).
  const bool use_vm = vector_major_for(bits, n_byte_groups);
#if defined(__x86_64__) || defined(_M_X64)
  if (use_vm) {
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
    const bool use_pd = false;
#else
    const bool use_pd = (bits == 4);
#endif
    prep.backend = use_pd ? SearchBackendKind::VmPermuteDot : SearchBackendKind::VmVnni;
    if (use_pd) {
      prep.pds.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        auto qrow =
            std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
        prep.pds[static_cast<std::size_t>(qi)] = build_permute_dot(qrow, centroids, dim);
      }
    } else {
      prep.split_luts.resize(nq);
      prep.lut_scales.resize(nq);
      prep.lut_biases.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        auto qrow =
            std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
        PreparedQueryLut lut = build_query_lut(qrow, centroids, bits, dim);
        prep.split_luts[static_cast<std::size_t>(qi)] =
            split_lut_for_vnni(lut.uint8_luts, n_byte_groups);
        prep.lut_scales[static_cast<std::size_t>(qi)] = lut.scale;
        prep.lut_biases[static_cast<std::size_t>(qi)] = lut.bias;
      }
    }
    return prep;
  }
  prep.backend = cpu_has_avx2() ? SearchBackendKind::Avx2 : SearchBackendKind::Scalar;
#else
  (void)use_vm;
  prep.backend = SearchBackendKind::Scalar;
#endif

  prep.luts.resize(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    auto qrow = std::span<const float>(prep.q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
    prep.luts[static_cast<std::size_t>(qi)] = build_query_lut(qrow, centroids, bits, dim);
  }
  return prep;
}

PreparedQueries prepared_query_at(const PreparedQueries& prep, std::size_t qi) {
  if (qi >= prep.nq) {
    throw std::out_of_range("prepared_query_at: qi out of range");
  }
  PreparedQueries one;
  one.nq = 1;
  one.dim = prep.dim;
  one.bits = prep.bits;
  one.n_byte_groups = prep.n_byte_groups;
  one.backend = prep.backend;
  if (!prep.q_rot.empty()) {
    one.q_rot.assign(prep.q_rot.begin() + static_cast<std::ptrdiff_t>(qi * prep.dim),
                     prep.q_rot.begin() + static_cast<std::ptrdiff_t>((qi + 1) * prep.dim));
  }
  if (!prep.bias_corrs.empty()) one.bias_corrs = {prep.bias_corrs[qi]};
  if (!prep.pds.empty()) one.pds = {prep.pds[qi]};
  if (!prep.split_luts.empty()) {
    one.split_luts = {prep.split_luts[qi]};
    one.lut_scales = {prep.lut_scales[qi]};
    one.lut_biases = {prep.lut_biases[qi]};
  }
  if (!prep.luts.empty()) one.luts = {prep.luts[qi]};
  return one;
}


SearchResults score_prepared_vm_threshold(const PreparedQueries& prep, std::size_t capacity,
                                          std::span<const std::uint8_t> blocked_codes,
                                          std::size_t n_blocks, std::span<const float> scales,
                                          std::span<const std::size_t> query_indices,
                                          float score_threshold,
                                          LandmarkStageTimings* timings = nullptr,
                                          const float* residual_norms = nullptr,
                                          const float* residual_scales = nullptr) {
  const std::size_t nq = query_indices.empty() ? prep.nq : query_indices.size();
  auto global_qi = [&](std::size_t local_i) -> std::size_t {
    return query_indices.empty() ? local_i : query_indices[local_i];
  };
  const std::size_t n_vectors = scales.size();
  const bool use_pd = prep.backend == SearchBackendKind::VmPermuteDot;
  const std::size_t n_byte_groups = prep.n_byte_groups;

  SearchResults out;
  out.nq = nq;
  out.k = capacity;
  out.scores.resize_uninitialized(nq * capacity);
  out.ids.resize_uninitialized(nq * capacity);
  out.counts.assign(nq, 0);
  (void)timings;
  if (nq == 0 || capacity == 0) return out;

#if defined(__x86_64__) || defined(_M_X64)
  std::size_t n_threads = 1;
#ifdef _OPENMP
  n_threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
  const std::size_t nq_batch = nq_batch_for(use_pd, n_threads, nq);

  // turbovec block-parallelizes a *single* query (bandwidth-bound on one core).
  // Multi-query already has query-batch tiles; merging large threshold buffers
  // across block ranges is slower than one codes pass per batch.
  const bool block_parallel = (nq == 1 && n_threads > 1 && single_query_parallelizes(n_vectors));
  const std::size_t n_quads = (nq + nq_batch - 1) / nq_batch;

  if (!block_parallel) {
    // Fast path: write straight into `out` (no merge copies).
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bi = 0; bi < static_cast<int>(n_quads); ++bi) {
      const std::size_t qi_start = static_cast<std::size_t>(bi) * nq_batch;
      const std::size_t batch_nq = std::min(nq_batch, nq - qi_start);
      float* score_ptrs[kNqBatchWide];
      std::uint64_t* id_ptrs[kNqBatchWide];
      std::size_t hit_counts[kNqBatchWide] = {};
      for (std::size_t i = 0; i < batch_nq; ++i) {
        score_ptrs[i] = out.scores.data() + (qi_start + i) * capacity;
        id_ptrs[i] = out.ids.data() + (qi_start + i) * capacity;
      }
      if (use_pd) {
        const QueryPermuteDot* pd_ptrs[kNqBatchWide];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          pd_ptrs[i] = &prep.pds[global_qi(qi_start + i)];
        }
        const float* rs_batch = nullptr;
        float rs_local[kNqBatchWide];
        if (residual_scales != nullptr) {
          for (std::size_t i = 0; i < batch_nq; ++i) {
            rs_local[i] = residual_scales[global_qi(qi_start + i)];
          }
          rs_batch = rs_local;
        }
        score_queries_permute_dot_threshold_fill(
            pd_ptrs, batch_nq, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks,
            score_threshold, capacity, score_ptrs, id_ptrs, hit_counts, nullptr, nullptr,
            residual_norms, rs_batch);
      } else {
        const std::uint8_t* lut_ptrs[kNqBatchWide];
        float lut_scales[kNqBatchWide];
        float lut_biases[kNqBatchWide];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          const std::size_t gqi = global_qi(qi_start + i);
          lut_ptrs[i] = prep.split_luts[gqi].data();
          lut_scales[i] = prep.lut_scales[gqi];
          lut_biases[i] = prep.lut_biases[gqi];
        }
        const float* rs_batch = nullptr;
        float rs_local[kNqBatchWide];
        if (residual_scales != nullptr) {
          for (std::size_t i = 0; i < batch_nq; ++i) {
            rs_local[i] = residual_scales[global_qi(qi_start + i)];
          }
          rs_batch = rs_local;
        }
        score_queries_vnni_threshold_fill(lut_ptrs, lut_scales, lut_biases, batch_nq, blocked_codes,
                                          scales, n_byte_groups, n_vectors, n_blocks,
                                          score_threshold, capacity, score_ptrs, id_ptrs,
                                          hit_counts, nullptr, nullptr, residual_norms, rs_batch);
      }
      for (std::size_t i = 0; i < batch_nq; ++i) {
        out.counts[qi_start + i] = hit_counts[i];
      }
    }
    return out;
  }

  // nq==1: turbovec `block_range_stride` — one large range per thread, then collect+sort.
  const std::size_t blocks_per_range = block_range_stride(n_blocks, n_threads);
  const std::size_t n_ranges = (n_blocks + blocks_per_range - 1) / blocks_per_range;
  std::vector<std::vector<TileCand>> per_range(n_ranges);
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel for schedule(static)
#endif
  for (int rs = 0; rs < static_cast<int>(n_ranges); ++rs) {
    const std::size_t block_start = static_cast<std::size_t>(rs) * blocks_per_range;
    if (block_start >= n_blocks) continue;
    const std::size_t block_end = std::min(block_start + blocks_per_range, n_blocks);
    const std::size_t range_blocks = block_end - block_start;
    const std::size_t vec_start = block_start * kBlock;
    const std::size_t range_n = std::min(block_end * kBlock, n_vectors) - vec_start;
    const std::size_t block_bytes = n_byte_groups * kBlock;
    auto codes_span = blocked_codes.subspan(block_start * block_bytes, range_blocks * block_bytes);
    auto scales_span = scales.subspan(vec_start, range_n);

    std::vector<float> scores(capacity);
    std::vector<std::uint64_t> ids(capacity);
    std::size_t hit = 0;
    float* sp = scores.data();
    std::uint64_t* ip = ids.data();
    const float* resid_range =
        residual_norms != nullptr ? residual_norms + vec_start : nullptr;
    const float rs0 = residual_scales != nullptr ? residual_scales[global_qi(0)] : 0.f;
    if (use_pd) {
      const QueryPermuteDot* pd = &prep.pds[global_qi(0)];
      score_queries_permute_dot_threshold_fill(&pd, 1, codes_span, scales_span, n_byte_groups,
                                               range_n, range_blocks, score_threshold, capacity,
                                               &sp, &ip, &hit, nullptr, nullptr, resid_range, &rs0);
    } else {
      const std::uint8_t* lp = prep.split_luts[global_qi(0)].data();
      const float ls = prep.lut_scales[global_qi(0)];
      const float lb = prep.lut_biases[global_qi(0)];
      score_queries_vnni_threshold_fill(&lp, &ls, &lb, 1, codes_span, scales_span, n_byte_groups,
                                        range_n, range_blocks, score_threshold, capacity, &sp, &ip,
                                        &hit, nullptr, nullptr, resid_range, &rs0);
    }
    auto& local = per_range[static_cast<std::size_t>(rs)];
    local.reserve(hit);
    for (std::size_t j = 0; j < hit; ++j) {
      local.push_back(TileCand{scores[j], ids[j] + static_cast<std::uint64_t>(vec_start)});
    }
  }

  std::vector<TileCand> merged;
  std::size_t total = 0;
  for (const auto& r : per_range) total += r.size();
  merged.reserve(total);
  for (auto& r : per_range) {
    merged.insert(merged.end(), r.begin(), r.end());
  }

  if (merged.size() > capacity) {
    std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(capacity),
                      merged.end(), [](const TileCand& a, const TileCand& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.id < b.id;
                      });
    merged.resize(capacity);
  } else {
    std::sort(merged.begin(), merged.end(), [](const TileCand& a, const TileCand& b) {
      if (a.score != b.score) return a.score > b.score;
      return a.id < b.id;
    });
  }
  out.counts[0] = merged.size();
  for (std::size_t j = 0; j < merged.size(); ++j) {
    out.scores[j] = merged[j].score;
    out.ids[j] = merged[j].id;
  }
  return out;
#else
  (void)blocked_codes;
  (void)n_blocks;
  (void)score_threshold;
  (void)use_pd;
  (void)n_byte_groups;
  (void)timings;
  throw std::logic_error("VM threshold shortlist requires x86_64");
#endif
}

SearchResults score_prepared_perm0_threshold(const PreparedQueries& prep, std::size_t capacity,
                                             std::span<const std::uint8_t> blocked_codes,
                                             std::size_t n_blocks, std::span<const float> scales,
                                             std::span<const std::size_t> query_indices,
                                             float score_threshold) {
  const std::size_t nq = query_indices.empty() ? prep.nq : query_indices.size();
  auto global_qi = [&](std::size_t local_i) -> std::size_t {
    return query_indices.empty() ? local_i : query_indices[local_i];
  };
  const std::size_t n_vectors = scales.size();
  const bool use_simd = prep.backend == SearchBackendKind::Avx2;
  const std::size_t n_byte_groups = prep.n_byte_groups;

  SearchResults out;
  out.nq = nq;
  out.k = capacity;
  out.scores.resize_uninitialized(nq * capacity);
  out.ids.resize_uninitialized(nq * capacity);
  out.counts.assign(nq, 0);
  if (nq == 0 || capacity == 0) return out;

  if (!use_simd) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t gqi = global_qi(static_cast<std::size_t>(qi));
      const auto& lut = prep.luts[gqi];
      const float bias_corr = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      std::size_t count = 0;
      for (std::size_t b = 0; b < n_blocks && count < capacity; ++b) {
        const std::size_t base_vec = b * kBlock;
        for (std::size_t lane = 0; lane < kBlock && count < capacity; ++lane) {
          const std::size_t vi = base_vec + lane;
          if (vi >= n_vectors) break;
          float score = lut.bias + bias_corr;
          for (std::size_t g = 0; g < n_byte_groups; ++g) {
            const auto byte_val = static_cast<std::size_t>(
                read_code(blocked_codes, prep.bits, n_byte_groups, b, g, lane));
            score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + (byte_val >> 4)]);
            score +=
                lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + 16 + (byte_val & 0x0F)]);
          }
          score *= scales[vi];
          if (score >= score_threshold) {
            out.scores[static_cast<std::size_t>(qi) * capacity + count] = score;
            out.ids[static_cast<std::size_t>(qi) * capacity + count] =
                static_cast<std::uint64_t>(vi);
            ++count;
          }
        }
      }
      out.counts[static_cast<std::size_t>(qi)] = count;
    }
    return out;
  }

#if defined(__x86_64__) || defined(_M_X64)
  std::size_t n_threads = 1;
#ifdef _OPENMP
  n_threads = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
  // turbovec nq=1 block-parallel covers AVX2/BW too.
  const bool block_parallel = (nq == 1 && n_threads > 1 && single_query_parallelizes(n_vectors));
  if (block_parallel) {
    const std::size_t blocks_per_range = block_range_stride(n_blocks, n_threads);
    const std::size_t n_ranges = (n_blocks + blocks_per_range - 1) / blocks_per_range;
    std::vector<std::vector<TileCand>> per_range(n_ranges);
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel for schedule(static)
#endif
    for (int rs = 0; rs < static_cast<int>(n_ranges); ++rs) {
      const std::size_t block_start = static_cast<std::size_t>(rs) * blocks_per_range;
      if (block_start >= n_blocks) continue;
      const std::size_t block_end = std::min(block_start + blocks_per_range, n_blocks);
      const std::size_t range_blocks = block_end - block_start;
      const std::size_t vec_start = block_start * kBlock;
      const std::size_t range_n = std::min(block_end * kBlock, n_vectors) - vec_start;
      const std::size_t block_bytes = n_byte_groups * kBlock;
      auto codes_span =
          blocked_codes.subspan(block_start * block_bytes, range_blocks * block_bytes);
      auto scales_span = scales.subspan(vec_start, range_n);
      const std::size_t gqi = global_qi(0);
      const auto& lut = prep.luts[gqi];
      QueryLutView view{lut.uint8_luts.data(), lut.scale, lut.bias};
      const float bc = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      std::vector<float> scores(capacity);
      std::vector<std::uint64_t> ids(capacity);
      std::size_t hit = 0;
      score_query_avx2_perm0_threshold_fill(view, codes_span, scales_span, n_byte_groups, range_n,
                                            range_blocks, score_threshold, capacity, scores.data(),
                                            ids.data(), &hit, bc, nullptr, 0.f);
      auto& local = per_range[static_cast<std::size_t>(rs)];
      local.reserve(hit);
      for (std::size_t j = 0; j < hit; ++j) {
        local.push_back(TileCand{scores[j], ids[j] + static_cast<std::uint64_t>(vec_start)});
      }
    }
    std::vector<TileCand> merged;
    std::size_t total = 0;
    for (const auto& r : per_range) total += r.size();
    merged.reserve(total);
    for (auto& r : per_range) merged.insert(merged.end(), r.begin(), r.end());
    if (merged.size() > capacity) {
      std::partial_sort(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(capacity),
                        merged.end(), [](const TileCand& a, const TileCand& b) {
                          if (a.score != b.score) return a.score > b.score;
                          return a.id < b.id;
                        });
      merged.resize(capacity);
    } else {
      std::sort(merged.begin(), merged.end(), [](const TileCand& a, const TileCand& b) {
        if (a.score != b.score) return a.score > b.score;
        return a.id < b.id;
      });
    }
    out.counts[0] = merged.size();
    for (std::size_t j = 0; j < merged.size(); ++j) {
      out.scores[j] = merged[j].score;
      out.ids[j] = merged[j].id;
    }
    return out;
  }

  // Query-batch tiles write directly into `out`.
  const std::size_t n_quads = (nq + kAvx2NqBatch - 1) / kAvx2NqBatch;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int bi = 0; bi < static_cast<int>(n_quads); ++bi) {
    const std::size_t qi_start = static_cast<std::size_t>(bi) * kAvx2NqBatch;
    const std::size_t batch_nq = std::min(kAvx2NqBatch, nq - qi_start);
    QueryLutView lut_views[kAvx2NqBatch];
    float bias_corrs[kAvx2NqBatch];
    float* score_ptrs[kAvx2NqBatch];
    std::uint64_t* id_ptrs[kAvx2NqBatch];
    std::size_t hit_counts[kAvx2NqBatch] = {};
    for (std::size_t i = 0; i < batch_nq; ++i) {
      const std::size_t gqi = global_qi(qi_start + i);
      const auto& lut = prep.luts[gqi];
      lut_views[i] = QueryLutView{lut.uint8_luts.data(), lut.scale, lut.bias};
      bias_corrs[i] = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      score_ptrs[i] = out.scores.data() + (qi_start + i) * capacity;
      id_ptrs[i] = out.ids.data() + (qi_start + i) * capacity;
    }
    score_queries_avx2_perm0_threshold_fill(
        lut_views, batch_nq, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks,
        score_threshold, capacity, score_ptrs, id_ptrs, hit_counts, bias_corrs, nullptr, nullptr);
    for (std::size_t i = 0; i < batch_nq; ++i) {
      out.counts[qi_start + i] = hit_counts[i];
    }
  }
  return out;
#else
  throw std::logic_error("AVX2 threshold shortlist requires x86_64");
#endif
}

SearchResults score_prepared(const PreparedQueries& prep, std::size_t k,
                             std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                             std::span<const float> scales, std::span<const std::uint64_t> id_map,
                             std::span<const std::size_t> query_indices, float score_threshold,
                             LandmarkStageTimings* timings, std::span<const float> residual_norms,
                             std::span<const float> residual_scales,
                             std::span<const float> scale_resid) {
  if (!query_indices.empty()) {
    for (std::size_t qi : query_indices) {
      if (qi >= prep.nq) {
        throw std::out_of_range("score_prepared: query_indices out of range");
      }
    }
  }
  const std::size_t nq_out = query_indices.empty() ? prep.nq : query_indices.size();
  const std::size_t n_vectors = scales.size();
  if (n_vectors == 0 || nq_out == 0) {
    SearchResults out;
    out.nq = nq_out;
    out.k = 0;
    return out;
  }
  if (!id_map.empty() && id_map.size() != n_vectors) {
    throw std::invalid_argument("id_map size must equal number of vectors in range");
  }
  if (!residual_norms.empty() && residual_norms.size() != n_vectors) {
    throw std::invalid_argument("residual_norms size must equal number of vectors");
  }
  if (!residual_scales.empty() && residual_scales.size() != prep.nq) {
    throw std::invalid_argument("residual_scales size must equal prep.nq");
  }
  if (!scale_resid.empty() && scale_resid.size() != 2 * n_vectors) {
    throw std::invalid_argument("scale_resid size must be 2 * number of vectors");
  }
  const float* resid_ptr = residual_norms.empty() ? nullptr : residual_norms.data();
  const float* resid_sc_ptr = residual_scales.empty() ? nullptr : residual_scales.data();
  const float* scale_resid_ptr = scale_resid.empty() ? nullptr : scale_resid.data();

  SearchResults out;
  if (std::isfinite(score_threshold)) {
    const std::size_t capacity = k == 0 ? n_vectors : std::min(k, n_vectors);
    if (prep.backend == SearchBackendKind::VmPermuteDot ||
        prep.backend == SearchBackendKind::VmVnni) {
      out = score_prepared_vm_threshold(prep, capacity, blocked_codes, n_blocks, scales,
                                        query_indices, score_threshold, timings, resid_ptr,
                                        resid_sc_ptr);
    } else {
      out = score_prepared_perm0_threshold(prep, capacity, blocked_codes, n_blocks, scales,
                                           query_indices, score_threshold);
      // AVX2 path: apply residual bound after fill (membership was s_L-only).
      if (resid_ptr != nullptr && resid_sc_ptr != nullptr) {
        for (std::size_t local = 0; local < out.nq; ++local) {
          const std::size_t gqi = query_indices.empty() ? local : query_indices[local];
          const float rs = resid_sc_ptr[gqi];
          const std::size_t n_cand =
              out.counts.empty() ? out.k : std::min(out.counts[local], out.k);
          for (std::size_t j = 0; j < n_cand; ++j) {
            const std::size_t vi = static_cast<std::size_t>(out.ids[local * out.k + j]);
            out.scores[local * out.k + j] += rs * resid_ptr[vi];
          }
        }
      }
      if (timings) {
      }
    }
  } else {
    const std::size_t effective_k = std::min(k, n_vectors);
    if (prep.backend == SearchBackendKind::VmPermuteDot ||
        prep.backend == SearchBackendKind::VmVnni) {
      out = score_prepared_vm(prep, effective_k, blocked_codes, n_blocks, scales, query_indices,
                              resid_ptr, resid_sc_ptr, timings, scale_resid_ptr);
    } else {
      out = score_prepared_perm0(prep, effective_k, blocked_codes, n_blocks, scales, query_indices);
      // AVX2/scalar heap: apply residual after (membership was s_L-only).
      if (resid_ptr != nullptr && resid_sc_ptr != nullptr) {
        for (std::size_t local = 0; local < out.nq; ++local) {
          const std::size_t gqi = query_indices.empty() ? local : query_indices[local];
          const float rs = resid_sc_ptr[gqi];
          for (std::size_t j = 0; j < out.k; ++j) {
            const std::size_t vi = static_cast<std::size_t>(out.ids[local * out.k + j]);
            if (vi < n_vectors) out.scores[local * out.k + j] += rs * resid_ptr[vi];
          }
          // Re-sort after residual correction.
          std::vector<std::size_t> order(out.k);
          for (std::size_t j = 0; j < out.k; ++j) order[j] = j;
          float* sc = out.scores.data() + local * out.k;
          std::uint64_t* id = out.ids.data() + local * out.k;
          std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
            if (sc[a] != sc[b]) return sc[a] > sc[b];
            return id[a] < id[b];
          });
          std::vector<float> sc_tmp(out.k);
          std::vector<std::uint64_t> id_tmp(out.k);
          for (std::size_t j = 0; j < out.k; ++j) {
            sc_tmp[j] = sc[order[j]];
            id_tmp[j] = id[order[j]];
          }
          std::copy(sc_tmp.begin(), sc_tmp.end(), sc);
          std::copy(id_tmp.begin(), id_tmp.end(), id);
        }
      }
      if (timings) {
      }
    }
  }
  remap_ids(out, id_map);
  return out;
}

void score_prepared_into(const PreparedQueries& prep, std::size_t k,
                         std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                         std::span<const float> scales, std::span<const std::uint64_t> id_map,
                         std::span<const std::size_t> query_indices, float* heap_s,
                         std::uint64_t* heap_i, std::size_t* heap_sz, float* heap_min,
                         std::size_t* heap_mi, const float* score_offsets) {
  if (query_indices.empty()) {
    throw std::invalid_argument("score_prepared_into requires non-empty query_indices");
  }
  for (std::size_t qi : query_indices) {
    if (qi >= prep.nq) {
      throw std::out_of_range("score_prepared_into: query_indices out of range");
    }
  }
  const std::size_t n_vectors = scales.size();
  if (n_vectors == 0 || k == 0) return;
  if (!id_map.empty() && id_map.size() != n_vectors) {
    throw std::invalid_argument("id_map size must equal number of vectors in range");
  }
  const std::uint64_t* id_ptr = id_map.empty() ? nullptr : id_map.data();
  const std::size_t nqi = query_indices.size();

  // Batch VNNI / permute-dot over the shared codes stream (up to kNqBatch queries).
  if (prep.backend == SearchBackendKind::VmVnni ||
      prep.backend == SearchBackendKind::VmPermuteDot) {
    const std::size_t n_batches = (nqi + kNqBatch - 1) / kNqBatch;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bi = 0; bi < static_cast<int>(n_batches); ++bi) {
      const std::size_t start = static_cast<std::size_t>(bi) * kNqBatch;
      const std::size_t batch_nq = std::min(kNqBatch, nqi - start);
      float* hs_ptrs[kNqBatch];
      std::uint64_t* hi_ptrs[kNqBatch];
      std::size_t hsz_local[kNqBatch];
      float hmin_local[kNqBatch];
      std::size_t hmi_local[kNqBatch];
      float offs_local[kNqBatch];

      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t gqi = query_indices[start + i];
        hs_ptrs[i] = heap_s + gqi * k;
        hi_ptrs[i] = heap_i + gqi * k;
        hsz_local[i] = heap_sz[gqi];
        hmin_local[i] = heap_min[gqi];
        hmi_local[i] = heap_mi[gqi];
        offs_local[i] = score_offsets ? score_offsets[gqi] : 0.f;
      }

      if (prep.backend == SearchBackendKind::VmVnni) {
        const std::uint8_t* lut_ptrs[kNqBatch];
        float scales_batch[kNqBatch];
        float biases_batch[kNqBatch];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          const std::size_t gqi = query_indices[start + i];
          lut_ptrs[i] = prep.split_luts[gqi].data();
          scales_batch[i] = prep.lut_scales[gqi];
          biases_batch[i] = prep.lut_biases[gqi];
        }
        score_queries_vnni(lut_ptrs, scales_batch, biases_batch, batch_nq, blocked_codes, scales,
                           prep.n_byte_groups, n_vectors, n_blocks, k, hs_ptrs, hi_ptrs, hsz_local,
                           hmin_local, hmi_local, id_ptr, offs_local);
      } else {
        const QueryPermuteDot* pd_ptrs[kNqBatch];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          pd_ptrs[i] = &prep.pds[query_indices[start + i]];
        }
        score_queries_permute_dot(pd_ptrs, batch_nq, blocked_codes, scales, prep.n_byte_groups,
                                  n_vectors, n_blocks, k, hs_ptrs, hi_ptrs, hsz_local, hmin_local,
                                  hmi_local, id_ptr, offs_local);
      }

      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t gqi = query_indices[start + i];
        heap_sz[gqi] = hsz_local[i];
        heap_min[gqi] = hmin_local[i];
        heap_mi[gqi] = hmi_local[i];
      }
    }
    return;
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int ii = 0; ii < static_cast<int>(nqi); ++ii) {
    const std::size_t gqi = query_indices[static_cast<std::size_t>(ii)];
    float* hs = heap_s + gqi * k;
    std::uint64_t* hi = heap_i + gqi * k;
    std::size_t& hsz = heap_sz[gqi];
    float& hmin = heap_min[gqi];
    std::size_t& hmi = heap_mi[gqi];
    const float off = score_offsets ? score_offsets[gqi] : 0.f;

#if defined(__x86_64__) || defined(_M_X64)
    if (prep.backend == SearchBackendKind::Avx2) {
      const auto& lut = prep.luts[gqi];
      QueryLutView view{lut.uint8_luts.data(), lut.scale, lut.bias};
      score_query_avx2_perm0(view, blocked_codes, scales, prep.n_byte_groups, n_vectors, n_blocks, k,
                             hs, hi, hsz, hmin, hmi, prep.bias_corrs[gqi], id_ptr, off);
    } else
#endif
    {
      score_query_scalar(prep.luts[gqi], blocked_codes, scales, prep.bits, prep.n_byte_groups,
                         n_vectors, n_blocks, k, hs, hi, hsz, hmin, hmi, prep.bias_corrs[gqi],
                         id_ptr, off);
    }
  }
}

void heaps_to_search_results(SearchResults& out, std::size_t nq, std::size_t k,
                             const float* heap_s, const std::uint64_t* heap_i,
                             const std::size_t* heap_sz) {
  out.nq = nq;
  out.k = k;
  out.scores.assign(nq * k, 0.f);
  out.ids.assign(nq * k, 0);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const std::size_t hsz = heap_sz[qi];
    std::vector<std::size_t> order(hsz);
    for (std::size_t i = 0; i < hsz; ++i) order[i] = i;
    const float* hs = heap_s + qi * k;
    const std::uint64_t* hi = heap_i + qi * k;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      if (hs[a] != hs[b]) return hs[a] > hs[b];
      return hi[a] < hi[b];
    });
    for (std::size_t i = 0; i < hsz; ++i) {
      out.scores[qi * k + i] = hs[order[i]];
      out.ids[qi * k + i] = hi[order[i]];
    }
  }
}

SearchResults search_flat(std::span<const float> queries, std::size_t nq, std::size_t dim,
                          std::size_t k, const Rotation& rotation,
                          std::span<const float> centroids, std::size_t bits,
                          std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                          std::span<const float> scales, std::span<const float> tqplus_shift,
                          std::span<const float> tqplus_scale) {
  auto prep =
      prepare_queries(queries, nq, dim, rotation, centroids, bits, tqplus_shift, tqplus_scale);
  return score_prepared(prep, k, blocked_codes, n_blocks, scales);
}

SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, std::size_t shortlist,
                                     std::span<const float> landmark_basis, std::size_t n_landmarks,
                                     std::size_t scan_dim, std::span<const float> lm_centroids,
                                     std::size_t scan_bits, std::span<const std::uint8_t> blocked_codes,
                                     std::size_t n_blocks, std::span<const float> lm_scales,
                                     std::span<const float> residual_norms, const Rotation& rotation,
                                     float score_threshold, LandmarkStageTimings* timings,
                                     bool shortlist_only, std::span<const float> scale_resid) {

  SearchResults out;
  out.nq = nq;
  out.k = k;
  out.scores.assign(nq * k, 0.f);
  out.ids.assign(nq * k, 0);
  if (nq == 0 || lm_scales.empty() || k == 0) {
    return out;
  }
  if (scan_dim < n_landmarks || scan_dim % 8 != 0) {
    throw std::invalid_argument("search_landmark_rerank: scan_dim must be >= m and multiple of 8");
  }
  if (scan_bits < 2 || scan_bits > 4) {
    throw std::invalid_argument("search_landmark_rerank: scan_bits must be 2, 3, or 4");
  }
  if (residual_norms.size() != lm_scales.size()) {
    throw std::invalid_argument("search_landmark_rerank: residual_norms size mismatch");
  }
  if (!scale_resid.empty() && scale_resid.size() != 2 * lm_scales.size()) {
    throw std::invalid_argument("search_landmark_rerank: scale_resid size mismatch");
  }
  const bool use_threshold = std::isfinite(score_threshold);
  if (use_threshold && shortlist == 0 && !shortlist_only) {
    throw std::invalid_argument(
        "search_landmark_rerank: shortlist is the mean target "
        "(buffer capacity = 2xshortlist) and must be > 0 when using score_threshold");
  }
  if (!use_threshold && shortlist_only) {
    throw std::invalid_argument(
        "search_landmark_rerank: shortlist_only requires finite score_threshold");
  }

  const std::size_t n_vectors = lm_scales.size();
  // Threshold mode: buffer = 2×shortlist. Heap mode: top-k heap (shortlist unused).
  const std::size_t sl =
      use_threshold ? (shortlist == 0 ? n_vectors : std::min(2 * shortlist, n_vectors))
                    : std::min(k, n_vectors);
  const std::size_t kk = std::min(k, n_vectors);

  std::vector<float> q_land(nq * scan_dim, 0.f);
  std::vector<float> q_rot(nq * dim);
  std::vector<float> q_perp(nq, 0.f);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    std::vector<float> scratch(dim);
    std::vector<float> row(dim);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      auto q = queries.subspan(qis * dim, dim);
      const float nrm = row_norm(q);
      const float inv = (nrm > kMinInputNorm) ? (1.f / nrm) : 0.f;
      rotation.apply_scaled_into(q, inv, row, scratch);
      std::copy(row.begin(), row.end(), q_rot.begin() + static_cast<std::ptrdiff_t>(qis * dim));
    }
  }
  if (dim % 8 != 0) {
    throw std::invalid_argument("search_landmark_rerank: dim must be multiple of 8");
  }
  if (landmark_basis.size() < n_landmarks * dim) {
    throw std::invalid_argument("search_landmark_rerank: landmark_basis too short");
  }
  project_onto_landmarks(q_rot.data(), nq, dim, landmark_basis.data(), n_landmarks, q_land.data(),
                         scan_dim);
  // ||q_perp||^2 = ||q_rot||^2 - ||q_L||^2 (orthonormal landmarks).
  for (std::size_t qi = 0; qi < nq; ++qi) {
    double q2 = 0.0;
    double l2 = 0.0;
    const float* qr = q_rot.data() + qi * dim;
    const float* ql = q_land.data() + qi * scan_dim;
    for (std::size_t d = 0; d < dim; ++d) q2 += static_cast<double>(qr[d]) * qr[d];
    for (std::size_t j = 0; j < n_landmarks; ++j) l2 += static_cast<double>(ql[j]) * ql[j];
    q_perp[qi] = static_cast<float>(std::sqrt(std::max(0.0, q2 - l2)));
  }
  auto prep = prepare_from_rows(q_land, nq, scan_dim, lm_centroids, scan_bits);

  if (shortlist_only && shortlist == 0) {
    // Count-only uncapped (tune-mean-shortlist): capacity 0 outs ⇒ hit counts only.
    std::vector<std::size_t> hit_counts(nq, 0);
#if defined(__x86_64__) || defined(_M_X64)
    if (prep.backend == SearchBackendKind::VmPermuteDot ||
        prep.backend == SearchBackendKind::VmVnni) {
      const bool use_pd = prep.backend == SearchBackendKind::VmPermuteDot;
      std::size_t n_threads_co = 1;
#ifdef _OPENMP
      n_threads_co = static_cast<std::size_t>(std::max(1, omp_get_max_threads()));
#endif
      const std::size_t nq_batch = nq_batch_for(use_pd, n_threads_co, nq);
      const std::size_t n_quads = (nq + nq_batch - 1) / nq_batch;
#if defined(_OPENMP) && !(defined(_WIN32) && defined(__GNUC__) && !defined(__clang__))
#pragma omp parallel for schedule(dynamic)
#endif
      for (int bi = 0; bi < static_cast<int>(n_quads); ++bi) {
        const std::size_t qi_start = static_cast<std::size_t>(bi) * nq_batch;
        const std::size_t batch_nq = std::min(nq_batch, nq - qi_start);
        std::size_t local_counts[kNqBatchWide] = {};
        float rs_local[kNqBatchWide];
        for (std::size_t i = 0; i < batch_nq; ++i) {
          rs_local[i] = q_perp[qi_start + i];
        }
        if (use_pd) {
          const QueryPermuteDot* pd_ptrs[kNqBatchWide];
          for (std::size_t i = 0; i < batch_nq; ++i) {
            pd_ptrs[i] = &prep.pds[qi_start + i];
          }
          score_queries_permute_dot_threshold_fill(
              pd_ptrs, batch_nq, blocked_codes, lm_scales, prep.n_byte_groups, n_vectors, n_blocks,
              score_threshold, /*capacity=*/0, /*out_scores=*/nullptr, /*out_ids=*/nullptr,
              local_counts, nullptr, nullptr, residual_norms.data(), rs_local);
        } else {
          const std::uint8_t* lut_ptrs[kNqBatchWide];
          float lut_scales[kNqBatchWide];
          float lut_biases[kNqBatchWide];
          for (std::size_t i = 0; i < batch_nq; ++i) {
            const std::size_t qi = qi_start + i;
            lut_ptrs[i] = prep.split_luts[qi].data();
            lut_scales[i] = prep.lut_scales[qi];
            lut_biases[i] = prep.lut_biases[qi];
          }
          score_queries_vnni_threshold_fill(
              lut_ptrs, lut_scales, lut_biases, batch_nq, blocked_codes, lm_scales,
              prep.n_byte_groups, n_vectors, n_blocks, score_threshold, /*capacity=*/0, nullptr,
              nullptr, local_counts, nullptr, nullptr, residual_norms.data(), rs_local);
        }
        for (std::size_t i = 0; i < batch_nq; ++i) {
          hit_counts[qi_start + i] = local_counts[i];
        }
      }
    } else
#endif
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
      for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
        const std::size_t qis = static_cast<std::size_t>(qi);
        std::size_t count = 0;
        const auto& lut = prep.luts[qis];
        const float rs = q_perp[qis];
        for (std::size_t b = 0; b < n_blocks; ++b) {
          const std::size_t base_vec = b * kBlock;
          for (std::size_t lane = 0; lane < kBlock; ++lane) {
            const std::size_t vi = base_vec + lane;
            if (vi >= n_vectors) break;
            float score = lut.bias;
            for (std::size_t g = 0; g < prep.n_byte_groups; ++g) {
              const auto byte_val = static_cast<std::size_t>(
                  read_code(blocked_codes, prep.bits, prep.n_byte_groups, b, g, lane));
              score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + (byte_val >> 4)]);
              score += lut.scale *
                       static_cast<float>(lut.uint8_luts[g * 32 + 16 + (byte_val & 0x0F)]);
            }
            score *= lm_scales[vi];
            score += rs * residual_norms[vi];
            if (score >= score_threshold) ++count;
          }
        }
        hit_counts[qis] = count;
      }
    }
    double mean_sl = 0.0;
    double max_sl = 0.0;
    for (std::size_t c : hit_counts) {
      mean_sl += static_cast<double>(c);
      max_sl = std::max(max_sl, static_cast<double>(c));
    }
    if (nq > 0) mean_sl /= static_cast<double>(nq);
    if (timings != nullptr) {
      timings->mean_shortlist = mean_sl;
      timings->max_shortlist = max_sl;
    }
    out.k = kk;
    return out;
  }
  SearchResults cand =
      score_prepared(prep, sl, blocked_codes, n_blocks, lm_scales, {}, {}, score_threshold,
                     timings ? timings : nullptr, residual_norms, q_perp, scale_resid);

  double mean_sl = 0.0;
  double max_sl = 0.0;
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const std::size_t n_cand =
        cand.counts.empty() ? cand.k : std::min(cand.counts[qi], cand.k);
    mean_sl += static_cast<double>(n_cand);
    max_sl = std::max(max_sl, static_cast<double>(n_cand));
  }
  if (nq > 0) mean_sl /= static_cast<double>(nq);

  if (!shortlist_only) {
    // Final top-k from residual-corrected landmark scores (no full-dim re-rank).
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::size_t n_cand =
          cand.counts.empty() ? cand.k : std::min(cand.counts[qis], cand.k);
      std::vector<std::size_t> order(n_cand);
      for (std::size_t j = 0; j < n_cand; ++j) order[j] = j;
      const float* cs = cand.scores.data() + qis * cand.k;
      const std::uint64_t* ci = cand.ids.data() + qis * cand.k;
      std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        if (cs[a] != cs[b]) return cs[a] > cs[b];
        return ci[a] < ci[b];
      });
      const std::size_t take = std::min(kk, n_cand);
      for (std::size_t j = 0; j < take; ++j) {
        out.scores[qis * kk + j] = cs[order[j]];
        out.ids[qis * kk + j] = ci[order[j]];
      }
      for (std::size_t j = take; j < kk; ++j) {
        out.scores[qis * kk + j] = 0.f;
        out.ids[qis * kk + j] = 0;
      }
    }
  }

  out.k = kk;
  if (timings != nullptr) {
    timings->mean_shortlist = mean_sl;
    timings->max_shortlist = max_sl;
  }
  return out;
}


void merge_search_results(SearchResults& dst, const SearchResults& src) {
  if (src.nq == 0 || src.k == 0) return;
  if (dst.nq == 0 || dst.k == 0) {
    dst = src;
    return;
  }
  if (dst.nq != src.nq) {
    throw std::invalid_argument("merge_search_results: nq mismatch");
  }
  struct Cand {
    float score;
    std::uint64_t id;
  };
  const std::size_t k = dst.k;
  for (std::size_t qi = 0; qi < dst.nq; ++qi) {
    std::vector<Cand> cands;
    cands.reserve(dst.k + src.k);
    for (std::size_t j = 0; j < dst.k; ++j) {
      cands.push_back(Cand{dst.scores[qi * dst.k + j], dst.ids[qi * dst.k + j]});
    }
    for (std::size_t j = 0; j < src.k; ++j) {
      cands.push_back(Cand{src.scores[qi * src.k + j], src.ids[qi * src.k + j]});
    }
    std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) {
      if (a.id != b.id) return a.id < b.id;
      return a.score > b.score;
    });
    std::vector<Cand> uniq;
    uniq.reserve(cands.size());
    for (const auto& c : cands) {
      if (uniq.empty() || uniq.back().id != c.id) {
        uniq.push_back(c);
      }
    }
    const std::size_t kk = std::min(k, uniq.size());
    std::partial_sort(uniq.begin(), uniq.begin() + static_cast<std::ptrdiff_t>(kk), uniq.end(),
                      [](const Cand& a, const Cand& b) {
                        if (a.score != b.score) return a.score > b.score;
                        return a.id < b.id;
                      });
    for (std::size_t j = 0; j < kk; ++j) {
      dst.scores[qi * k + j] = uniq[j].score;
      dst.ids[qi * k + j] = uniq[j].id;
    }
    for (std::size_t j = kk; j < k; ++j) {
      dst.scores[qi * k + j] = 0.f;
      dst.ids[qi * k + j] = 0;
    }
  }
}

}  // namespace vectorcache
