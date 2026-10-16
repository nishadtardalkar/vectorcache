#include "vectorcache/search/search.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/encode/encode.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/landmark_scan.hpp"
#include "vectorcache/search/search_avx2.hpp"
#include "vectorcache/search/search_avx512bw.hpp"
#include "vectorcache/search/topk_heap.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {
namespace {

constexpr std::size_t kTilesPerThread = 32;
constexpr std::size_t kMinTileBlocks = 1024;
constexpr std::size_t kMinTileBlocksX86 = kMinTileBlocks * 3;
constexpr std::size_t kSingleQueryParallelMinBlocks = 1024;

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

SearchResults score_prepared_perm0(const PreparedQueries& prep, std::size_t effective_k,
                                   std::span<const std::uint8_t> blocked_codes,
                                   std::size_t n_blocks, std::span<const float> scales,
                                   std::span<const std::size_t> query_indices) {
  const std::size_t nq = query_indices.empty() ? prep.nq : query_indices.size();
  auto global_qi = [&](std::size_t local_i) -> std::size_t {
    return query_indices.empty() ? local_i : query_indices[local_i];
  };
  const std::size_t n_vectors = scales.size();
  const bool use_simd = prep.backend == SearchBackendKind::Avx512bw ||
                       prep.backend == SearchBackendKind::Avx2;
  const bool use_avx512bw = prep.backend == SearchBackendKind::Avx512bw;
  const std::size_t n_byte_groups = prep.n_byte_groups;

  SearchResults out;
  out.nq = nq;
  out.k = effective_k;
  out.scores.assign(nq * effective_k, 0.f);
  out.ids.assign(nq * effective_k, 0);
  if (nq == 0) return out;

  // Scalar / non-x86: one query at a time.
  if (!use_simd) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t gqi = global_qi(static_cast<std::size_t>(qi));
      const auto& lut = prep.luts[gqi];
      std::vector<float> heap_s(effective_k);
      std::vector<std::uint64_t> heap_i(effective_k);
      std::size_t heap_sz = 0;
      float heap_min = 0.f;
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
      std::vector<float> heap_min(batch_nq, 0.f);
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

      if (use_avx512bw) {
        score_queries_avx512bw_perm0(lut_views, batch_nq, codes_span, scales_span, n_byte_groups,
                                     range_n, range_blocks, effective_k, heap_s_ptrs, heap_i_ptrs,
                                     heap_sz.data(), heap_min.data(), heap_mi.data(), bias_corrs,
                                     nullptr, nullptr);
      } else {
        score_queries_avx2_perm0(lut_views, batch_nq, codes_span, scales_span, n_byte_groups,
                                 range_n, range_blocks, effective_k, heap_s_ptrs, heap_i_ptrs,
                                 heap_sz.data(), heap_min.data(), heap_mi.data(), bias_corrs,
                                 nullptr, nullptr);
      }

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
  const bool use_simd = prep.backend == SearchBackendKind::Avx512bw ||
                       prep.backend == SearchBackendKind::Avx2;
  const bool use_avx512bw = prep.backend == SearchBackendKind::Avx512bw;
  const std::size_t n_byte_groups = prep.n_byte_groups;
  const std::size_t cap = capacity == 0 ? n_vectors : std::min(capacity, n_vectors);

  SearchResults out;
  out.nq = nq;
  out.k = cap;
  out.counts.assign(nq, 0);
  out.scores.assign(nq * cap, 0.f);
  out.ids.assign(nq * cap, 0);
  if (nq == 0 || cap == 0) {
    out.k = 0;
    return out;
  }

  if (!use_simd) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::size_t gqi = global_qi(qis);
      const auto& lut = prep.luts[gqi];
      const float bias_corr = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      float* row_s = out.scores.data() + qis * cap;
      std::uint64_t* row_i = out.ids.data() + qis * cap;
      std::size_t count = 0;
      for (std::size_t b = 0; b < n_blocks && count < cap; ++b) {
        const std::size_t base_vec = b * kBlock;
        for (std::size_t lane = 0; lane < kBlock && count < cap; ++lane) {
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
            row_s[count] = score;
            row_i[count] = static_cast<std::uint64_t>(vi);
            ++count;
          }
        }
      }
      out.counts[qis] = count;
    }
    return out;
  }

#if defined(__x86_64__) || defined(_M_X64)
  // turbovec-style: fuse NQ<=4 so each block's codes are loaded once per batch.
  // Parallelize over query batches (each batch owns its output rows → no races).
  const std::size_t n_quads = (nq + kAvx2NqBatch - 1) / kAvx2NqBatch;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int quad = 0; quad < static_cast<int>(n_quads); ++quad) {
    const std::size_t qi_start = static_cast<std::size_t>(quad) * kAvx2NqBatch;
    const std::size_t qi_end = std::min(qi_start + kAvx2NqBatch, nq);
    const std::size_t batch_nq = qi_end - qi_start;

    QueryLutView lut_views[kAvx2NqBatch];
    float bias_corrs[kAvx2NqBatch];
    float* score_ptrs[kAvx2NqBatch];
    std::uint64_t* id_ptrs[kAvx2NqBatch];
    std::size_t counts[kAvx2NqBatch] = {};
    for (std::size_t i = 0; i < batch_nq; ++i) {
      const std::size_t gqi = global_qi(qi_start + i);
      const auto& lut = prep.luts[gqi];
      lut_views[i] = QueryLutView{lut.uint8_luts.data(), lut.scale, lut.bias};
      bias_corrs[i] = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
      score_ptrs[i] = out.scores.data() + (qi_start + i) * cap;
      id_ptrs[i] = out.ids.data() + (qi_start + i) * cap;
    }

    if (use_avx512bw) {
      score_queries_avx512bw_perm0_threshold_fill(
          lut_views, batch_nq, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks,
          score_threshold, cap, score_ptrs, id_ptrs, counts, bias_corrs, nullptr, nullptr);
    } else {
      score_queries_avx2_perm0_threshold_fill(
          lut_views, batch_nq, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks,
          score_threshold, cap, score_ptrs, id_ptrs, counts, bias_corrs, nullptr, nullptr);
    }

    for (std::size_t i = 0; i < batch_nq; ++i) {
      out.counts[qi_start + i] = counts[i];
    }
  }
#else
  (void)n_byte_groups;
  throw std::logic_error("AVX2 threshold shortlist requires x86_64");
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

#if defined(__x86_64__) || defined(_M_X64)
  if (use_avx512bw()) {
    prep.backend = SearchBackendKind::Avx512bw;
  } else if (cpu_has_avx2()) {
    prep.backend = SearchBackendKind::Avx2;
  } else {
    prep.backend = SearchBackendKind::Scalar;
  }
#else
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
  // Same backend/LUT selection as prepare_queries, but rows are already transformed.
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;

  PreparedQueries prep;
  prep.nq = nq;
  prep.dim = dim;
  prep.bits = bits;
  prep.n_byte_groups = n_byte_groups;
  prep.q_rot.assign(rows.begin(), rows.begin() + static_cast<std::ptrdiff_t>(nq * dim));
  prep.bias_corrs.assign(nq, 0.f);

#if defined(__x86_64__) || defined(_M_X64)
  if (use_avx512bw()) {
    prep.backend = SearchBackendKind::Avx512bw;
  } else if (cpu_has_avx2()) {
    prep.backend = SearchBackendKind::Avx2;
  } else {
    prep.backend = SearchBackendKind::Scalar;
  }
#else
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
  if (!prep.luts.empty()) one.luts = {prep.luts[qi]};
  return one;
}

SearchResults score_prepared(const PreparedQueries& prep, std::size_t k,
                             std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                             std::span<const float> scales, std::span<const std::uint64_t> id_map,
                             std::span<const std::size_t> query_indices, float score_threshold) {
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

  SearchResults out;
  if (std::isfinite(score_threshold)) {
    // `k` is the fixed buffer capacity (typically 2× mean shortlist target).
    const std::size_t capacity = k == 0 ? n_vectors : std::min(k, n_vectors);
    out = score_prepared_perm0_threshold(prep, capacity, blocked_codes, n_blocks, scales,
                                         query_indices, score_threshold);
  } else {
    const std::size_t effective_k = std::min(k, n_vectors);
    out = score_prepared_perm0(prep, effective_k, blocked_codes, n_blocks, scales, query_indices);
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

  // Fused PERM0 batch (AVX-512BW or AVX2), nq ≤ 4.
  if (prep.backend == SearchBackendKind::Avx512bw || prep.backend == SearchBackendKind::Avx2) {
    const std::size_t batch_max = kAvx2NqBatch;
    const std::size_t n_batches = (nqi + batch_max - 1) / batch_max;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bi = 0; bi < static_cast<int>(n_batches); ++bi) {
      const std::size_t start = static_cast<std::size_t>(bi) * batch_max;
      const std::size_t batch_nq = std::min(batch_max, nqi - start);
      float* hs_ptrs[kAvx2NqBatch];
      std::uint64_t* hi_ptrs[kAvx2NqBatch];
      std::size_t hsz_local[kAvx2NqBatch];
      float hmin_local[kAvx2NqBatch];
      std::size_t hmi_local[kAvx2NqBatch];
      float offs_local[kAvx2NqBatch];
      float bias_corrs[kAvx2NqBatch];
      QueryLutView lut_views[kAvx2NqBatch];

      for (std::size_t i = 0; i < batch_nq; ++i) {
        const std::size_t gqi = query_indices[start + i];
        hs_ptrs[i] = heap_s + gqi * k;
        hi_ptrs[i] = heap_i + gqi * k;
        hsz_local[i] = heap_sz[gqi];
        hmin_local[i] = heap_min[gqi];
        hmi_local[i] = heap_mi[gqi];
        offs_local[i] = score_offsets ? score_offsets[gqi] : 0.f;
        bias_corrs[i] = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[gqi];
        const auto& lut = prep.luts[gqi];
        lut_views[i] = QueryLutView{lut.uint8_luts.data(), lut.scale, lut.bias};
      }

      if (prep.backend == SearchBackendKind::Avx512bw) {
        score_queries_avx512bw_perm0(lut_views, batch_nq, blocked_codes, scales, prep.n_byte_groups,
                                     n_vectors, n_blocks, k, hs_ptrs, hi_ptrs, hsz_local, hmin_local,
                                     hmi_local, bias_corrs, id_ptr, offs_local);
      } else {
        score_queries_avx2_perm0(lut_views, batch_nq, blocked_codes, scales, prep.n_byte_groups,
                                 n_vectors, n_blocks, k, hs_ptrs, hi_ptrs, hsz_local, hmin_local,
                                 hmi_local, bias_corrs, id_ptr, offs_local);
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
                                     std::span<const std::uint8_t> rerank_codes,
                                     std::span<const float> rerank_scales,
                                     std::span<const float> centroids, std::size_t rerank_bits,
                                     const Rotation& rotation, float score_threshold,
                                     LandmarkStageTimings* timings, bool shortlist_only) {
  using clock = std::chrono::steady_clock;
  auto ms_since = [](clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  };

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
  if (rerank_scales.size() != lm_scales.size()) {
    throw std::invalid_argument("search_landmark_rerank: rerank_scales size mismatch");
  }
  if (scan_bits < 2 || scan_bits > 4 || rerank_bits < 2 || rerank_bits > 4) {
    throw std::invalid_argument("search_landmark_rerank: scan/rerank bits must be 2, 3, or 4");
  }
  if (!std::isfinite(score_threshold)) {
    throw std::invalid_argument(
        "search_landmark_rerank: finite score_threshold required (no top-S heap shortlist)");
  }
  if (shortlist == 0 && !shortlist_only) {
    throw std::invalid_argument(
        "search_landmark_rerank: shortlist is the mean target "
        "(buffer capacity = 2×shortlist) and must be > 0");
  }
  const std::size_t codes_per_byte = 8 / rerank_bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  if (rerank_codes.size() < rerank_scales.size() * n_byte_groups) {
    throw std::invalid_argument("search_landmark_rerank: rerank_codes too short");
  }

  const std::size_t n_vectors = lm_scales.size();
  // `shortlist` = mean target; fixed buffer = 2× (stop when full). 0 = uncapped (tune).
  const std::size_t sl = shortlist == 0 ? n_vectors : std::min(2 * shortlist, n_vectors);
  const std::size_t kk = std::min(k, n_vectors);

  std::vector<float> q_land(nq * scan_dim, 0.f);
  std::vector<float> q_rot(nq * dim);

  auto t0 = clock::now();
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
      float* dest = q_land.data() + qis * scan_dim;
      for (std::size_t j = 0; j < n_landmarks; ++j) {
        double s = 0.0;
        const float* Lj = landmark_basis.data() + j * dim;
        for (std::size_t i = 0; i < dim; ++i) {
          s += static_cast<double>(row[i]) * Lj[i];
        }
        dest[j] = static_cast<float>(s);
      }
    }
  }
  const double project_ms = ms_since(t0);

  t0 = clock::now();
  auto prep = prepare_from_rows(q_land, nq, scan_dim, lm_centroids, scan_bits);
  const double shortlist_prep_ms = ms_since(t0);

  // Tune path: count threshold hits only (no candidate buffers).
  if (shortlist_only && shortlist == 0) {
    t0 = clock::now();
    std::vector<std::size_t> hit_counts(nq, 0);
    const std::size_t n_batches = (nq + kAvx2NqBatch - 1) / kAvx2NqBatch;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int bi = 0; bi < static_cast<int>(n_batches); ++bi) {
      const std::size_t qi_start = static_cast<std::size_t>(bi) * kAvx2NqBatch;
      const std::size_t batch_nq = std::min(kAvx2NqBatch, nq - qi_start);
      QueryLutView lut_views[kAvx2NqBatch];
      float bias_corrs[kAvx2NqBatch];
      std::size_t counts[kAvx2NqBatch] = {};
      for (std::size_t i = 0; i < batch_nq; ++i) {
        const auto& lut = prep.luts[qi_start + i];
        lut_views[i] = QueryLutView{lut.uint8_luts.data(), lut.scale, lut.bias};
        bias_corrs[i] = prep.bias_corrs.empty() ? 0.f : prep.bias_corrs[qi_start + i];
      }
#if defined(__x86_64__) || defined(_M_X64)
      if (prep.backend == SearchBackendKind::Avx512bw) {
        score_queries_avx512bw_perm0_threshold_fill(
            lut_views, batch_nq, blocked_codes, lm_scales, prep.n_byte_groups, n_vectors, n_blocks,
            score_threshold, /*capacity=*/0, /*out_scores=*/nullptr, /*out_ids=*/nullptr, counts,
            bias_corrs, nullptr, nullptr);
      } else if (prep.backend == SearchBackendKind::Avx2) {
        // Single-query API supports count-only (capacity=0); multi does not.
        for (std::size_t i = 0; i < batch_nq; ++i) {
          score_query_avx2_perm0_threshold_fill(
              lut_views[i], blocked_codes, lm_scales, prep.n_byte_groups, n_vectors, n_blocks,
              score_threshold, /*capacity=*/0, /*out_scores=*/nullptr, /*out_ids=*/nullptr,
              &counts[i], bias_corrs[i], nullptr, 0.f);
        }
      } else
#endif
      {
        for (std::size_t i = 0; i < batch_nq; ++i) {
          const auto& lut = prep.luts[qi_start + i];
          const float bias_corr = bias_corrs[i];
          std::size_t count = 0;
          for (std::size_t b = 0; b < n_blocks; ++b) {
            const std::size_t base_vec = b * kBlock;
            for (std::size_t lane = 0; lane < kBlock; ++lane) {
              const std::size_t vi = base_vec + lane;
              if (vi >= n_vectors) break;
              float score = lut.bias + bias_corr;
              for (std::size_t g = 0; g < prep.n_byte_groups; ++g) {
                const auto byte_val = static_cast<std::size_t>(
                    read_code(blocked_codes, prep.bits, prep.n_byte_groups, b, g, lane));
                score += lut.scale * static_cast<float>(lut.uint8_luts[g * 32 + (byte_val >> 4)]);
                score += lut.scale *
                         static_cast<float>(lut.uint8_luts[g * 32 + 16 + (byte_val & 0x0F)]);
              }
              score *= lm_scales[vi];
              if (score >= score_threshold) ++count;
            }
          }
          counts[i] = count;
        }
      }
      for (std::size_t i = 0; i < batch_nq; ++i) {
        hit_counts[qi_start + i] = counts[i];
      }
    }
    const double shortlist_scan_ms = ms_since(t0);
    double mean_sl = 0.0;
    double max_sl = 0.0;
    for (std::size_t c : hit_counts) {
      mean_sl += static_cast<double>(c);
      max_sl = std::max(max_sl, static_cast<double>(c));
    }
    if (nq > 0) mean_sl /= static_cast<double>(nq);
    if (timings != nullptr) {
      timings->project_ms = project_ms;
      timings->shortlist_prep_ms = shortlist_prep_ms;
      timings->shortlist_scan_ms = shortlist_scan_ms;
      timings->rerank_prep_ms = 0;
      timings->rerank_ms = 0;
      timings->mean_shortlist = mean_sl;
      timings->max_shortlist = max_sl;
    }
    out.k = kk;
    return out;
  }

  t0 = clock::now();
  SearchResults cand =
      score_prepared(prep, sl, blocked_codes, n_blocks, lm_scales, {}, {}, score_threshold);
  const double shortlist_scan_ms = ms_since(t0);

  double mean_sl = 0.0;
  double max_sl = 0.0;
  for (std::size_t qi = 0; qi < nq; ++qi) {
    const std::size_t n_cand =
        cand.counts.empty() ? cand.k : std::min(cand.counts[qi], cand.k);
    mean_sl += static_cast<double>(n_cand);
    max_sl = std::max(max_sl, static_cast<double>(n_cand));
  }
  if (nq > 0) mean_sl /= static_cast<double>(nq);

  double rerank_prep_ms = 0;
  double rerank_ms = 0;
  if (!shortlist_only) {
    // Flat-code re-rank needs uint8 LUTs (not vector-major scan prep).
    t0 = clock::now();
    std::vector<PreparedQueryLut> rerank_luts(nq);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      auto qrow =
          std::span<const float>(q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
      rerank_luts[static_cast<std::size_t>(qi)] =
          build_query_lut(qrow, centroids, rerank_bits, dim);
    }
    rerank_prep_ms = ms_since(t0);

    t0 = clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
      const std::size_t qis = static_cast<std::size_t>(qi);
      const std::size_t n_cand =
          cand.counts.empty() ? cand.k : std::min(cand.counts[qis], cand.k);
      if (n_cand == 0) {
        continue;
      }
      const auto& lut = rerank_luts[qis];
      quantized_rerank_topk(lut.uint8_luts, lut.scale, lut.bias, rerank_codes, n_byte_groups,
                            rerank_scales,
                            std::span<const std::uint64_t>(cand.ids.data() + qis * cand.k, n_cand),
                            n_cand, kk, out.scores.data() + qis * kk, out.ids.data() + qis * kk);
    }
    rerank_ms = ms_since(t0);
  }

  if (timings != nullptr) {
    timings->project_ms = project_ms;
    timings->shortlist_prep_ms = shortlist_prep_ms;
    timings->shortlist_scan_ms = shortlist_scan_ms;
    timings->rerank_prep_ms = rerank_prep_ms;
    timings->rerank_ms = rerank_ms;
    timings->mean_shortlist = mean_sl;
    timings->max_shortlist = max_sl;
  }

  out.k = kk;
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
