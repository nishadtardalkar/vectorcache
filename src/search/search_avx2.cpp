#include "vectorcache/search/search_avx2.hpp"

#include "vectorcache/pack/pack.hpp"

namespace vectorcache {
namespace {

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

}  // namespace

void score_query_avx2_perm0(QueryLutView lut, std::span<const std::uint8_t> blocked_codes,
                            std::span<const float> vec_scales, std::size_t n_byte_groups,
                            std::size_t n_vectors, std::size_t n_blocks, std::size_t k,
                            float* heap_s, std::uint64_t* heap_i, std::size_t& heap_sz,
                            float& heap_min, std::size_t& heap_mi, float bias_corr) {
  float scores[kBlock];
  for (std::size_t b = 0; b < n_blocks; ++b) {
    const std::size_t base_vec = b * kBlock;
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      scores[lane] = lut.bias + bias_corr;
    }
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      const std::size_t group_off = (b * n_byte_groups + g) * kBlock;
      const std::uint8_t* hi_plane = blocked_codes.data() + group_off;
      const std::uint8_t* lo_plane = blocked_codes.data() + group_off + 16;
      const std::uint8_t* hi_lut = lut.uint8_luts + g * 32;
      const std::uint8_t* lo_lut = lut.uint8_luts + g * 32 + 16;
      for (std::size_t j = 0; j < 16; ++j) {
        const std::size_t lane_a = kPerm0[j];
        const std::size_t lane_b = kPerm0[j] + 16;
        const std::uint8_t hp = hi_plane[j];
        const std::uint8_t lp = lo_plane[j];
        scores[lane_a] += lut.scale * static_cast<float>(hi_lut[hp & 0x0F]);
        scores[lane_a] += lut.scale * static_cast<float>(lo_lut[lp & 0x0F]);
        scores[lane_b] += lut.scale * static_cast<float>(hi_lut[hp >> 4]);
        scores[lane_b] += lut.scale * static_cast<float>(lo_lut[lp >> 4]);
      }
    }
    for (std::size_t lane = 0; lane < kBlock; ++lane) {
      const std::size_t vi = base_vec + lane;
      if (vi >= n_vectors) break;
      heap_push_or_replace(heap_s, heap_i, heap_sz, heap_min, heap_mi, k,
                           scores[lane] * vec_scales[vi], static_cast<std::uint64_t>(vi));
    }
  }
}

}  // namespace vectorcache
