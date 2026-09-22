#include "vectorcache/search/search.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "vectorcache/constants.hpp"
#include "vectorcache/encode/encode.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/search_avx2.hpp"
#include "vectorcache/search/search_vnni.hpp"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#endif

namespace vectorcache {
namespace {

struct QueryLut {
  std::vector<std::uint8_t> uint8_luts;
  float scale = 1.f;
  float bias = 0.f;
};

QueryLut build_query_lut(std::span<const float> q_rot_row, std::span<const float> centroids,
                         std::size_t bits, std::size_t dim) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t codes_per_nibble = codes_per_byte / 2;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  const std::uint16_t code_mask = static_cast<std::uint16_t>((1u << bits) - 1);

  QueryLut out;
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

void heap_push_or_replace(float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                          float& heap_min, std::size_t& heap_mi, std::size_t k, float score,
                          std::uint64_t id) {
  if (heap_sz < k) {
    heap_s[heap_sz] = score;
    heap_i[heap_sz] = id;
    ++heap_sz;
    if (heap_sz == k) {
      heap_min = heap_s[0];
      heap_mi = 0;
      for (std::size_t i = 1; i < k; ++i) {
        if (heap_s[i] < heap_min) {
          heap_min = heap_s[i];
          heap_mi = i;
        }
      }
    }
  } else if (score > heap_min) {
    heap_s[heap_mi] = score;
    heap_i[heap_mi] = id;
    heap_min = heap_s[0];
    heap_mi = 0;
    for (std::size_t i = 1; i < k; ++i) {
      if (heap_s[i] < heap_min) {
        heap_min = heap_s[i];
        heap_mi = i;
      }
    }
  }
}

void score_query_scalar(const QueryLut& lut, std::span<const std::uint8_t> blocked_codes,
                        std::span<const float> vec_scales, std::size_t bits,
                        std::size_t n_byte_groups, std::size_t n_vectors, std::size_t n_blocks,
                        std::size_t k, float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                        float& heap_min, std::size_t& heap_mi, float bias_corr) {
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
      score *= vec_scales[vi];
      heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k, score,
                           static_cast<std::uint64_t>(vi));
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

}  // namespace

SearchResults search_flat(std::span<const float> queries, std::size_t nq, std::size_t dim,
                          std::size_t k, const Rotation& rotation,
                          std::span<const float> centroids, std::size_t bits,
                          std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                          std::span<const float> scales, std::span<const float> tqplus_shift,
                          std::span<const float> tqplus_scale) {
  const std::size_t n_vectors = scales.size();
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  const std::size_t effective_k = std::min(k, n_vectors);

  SearchResults out;
  out.nq = nq;
  out.k = effective_k;
  out.scores.assign(nq * effective_k, 0.f);
  out.ids.assign(nq * effective_k, 0);

  std::vector<float> q_rot(nq * dim);
  std::vector<float> bias_corrs(nq, 0.f);

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
      auto dest = std::span<float>(q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
      if (!tqplus_shift.empty()) {
        double bc = 0.0;
        for (std::size_t d = 0; d < dim; ++d) {
          dest[d] = row[d] / tqplus_scale[d];
          bc -= static_cast<double>(row[d]) * static_cast<double>(tqplus_shift[d]);
        }
        bias_corrs[static_cast<std::size_t>(qi)] = static_cast<float>(bc);
      } else {
        std::copy(row.begin(), row.end(), dest.begin());
      }
    }
  }

  const bool use_vm = vector_major_for(bits, n_byte_groups);
#if defined(__x86_64__) || defined(_M_X64)
  const bool use_avx2 = !use_vm && cpu_has_avx2();
#else
  const bool use_avx2 = false;
  (void)use_vm;
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (int qi = 0; qi < static_cast<int>(nq); ++qi) {
    auto qrow = std::span<const float>(q_rot.data() + static_cast<std::size_t>(qi) * dim, dim);
    QueryLut lut = build_query_lut(qrow, centroids, bits, dim);
    std::vector<float> heap_s(effective_k);
    std::vector<std::uint64_t> heap_i(effective_k);
    std::size_t heap_sz = 0;
    float heap_min = 0.f;
    std::size_t heap_mi = 0;
#if defined(__x86_64__) || defined(_M_X64)
    if (use_vm) {
      auto split = split_lut_for_vnni(lut.uint8_luts, n_byte_groups);
      QueryLutView view{split.data(), lut.scale, lut.bias};
      score_query_vnni(view, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks, effective_k,
                       heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi,
                       bias_corrs[static_cast<std::size_t>(qi)]);
    } else if (use_avx2) {
      QueryLutView view{lut.uint8_luts.data(), lut.scale, lut.bias};
      score_query_avx2_perm0(view, blocked_codes, scales, n_byte_groups, n_vectors, n_blocks,
                             effective_k, heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi,
                             bias_corrs[static_cast<std::size_t>(qi)]);
    } else
#endif
    {
      score_query_scalar(lut, blocked_codes, scales, bits, n_byte_groups, n_vectors, n_blocks,
                         effective_k, heap_s.data(), heap_i.data(), heap_sz, heap_min, heap_mi,
                         bias_corrs[static_cast<std::size_t>(qi)]);
    }

    std::vector<std::size_t> order(heap_sz);
    for (std::size_t i = 0; i < heap_sz; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return heap_s[a] > heap_s[b]; });
    for (std::size_t i = 0; i < heap_sz; ++i) {
      out.scores[static_cast<std::size_t>(qi) * effective_k + i] = heap_s[order[i]];
      out.ids[static_cast<std::size_t>(qi) * effective_k + i] = heap_i[order[i]];
    }
  }

  return out;
}

}  // namespace vectorcache
