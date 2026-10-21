#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "vectorcache/search/search_vnni.hpp"
#include "vectorcache/transform/landmarks.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

/// Contiguous buffer that can grow without value-initializing.
template <typename T>
struct UninitBuffer {
  std::unique_ptr<T[]> data_;
  std::size_t size_ = 0;

  UninitBuffer() = default;
  UninitBuffer(UninitBuffer&&) noexcept = default;
  UninitBuffer& operator=(UninitBuffer&&) noexcept = default;
  UninitBuffer(const UninitBuffer& other) { *this = other; }
  UninitBuffer& operator=(const UninitBuffer& other) {
    if (this == &other) return *this;
    if (other.size_ == 0) {
      clear();
      return *this;
    }
    data_ = std::make_unique_for_overwrite<T[]>(other.size_);
    size_ = other.size_;
    std::copy_n(other.data_.get(), other.size_, data_.get());
    return *this;
  }

  void clear() noexcept {
    data_.reset();
    size_ = 0;
  }

  /// Allocate `n` elements; contents are indeterminate (trivial T only).
  void resize_uninitialized(std::size_t n) {
    if (n == 0) {
      clear();
      return;
    }
    data_ = std::make_unique_for_overwrite<T[]>(n);
    size_ = n;
  }

  void assign(std::size_t n, const T& value) {
    if (n == 0) {
      clear();
      return;
    }
    data_ = std::make_unique_for_overwrite<T[]>(n);
    size_ = n;
    std::fill_n(data_.get(), n, value);
  }

  T* data() noexcept { return data_.get(); }
  const T* data() const noexcept { return data_.get(); }
  T* begin() noexcept { return data_.get(); }
  const T* begin() const noexcept { return data_.get(); }
  T* end() noexcept { return data_.get() + size_; }
  const T* end() const noexcept { return data_.get() + size_; }
  std::size_t size() const noexcept { return size_; }
  bool empty() const noexcept { return size_ == 0; }
  T& operator[](std::size_t i) noexcept { return data_[i]; }
  const T& operator[](std::size_t i) const noexcept { return data_[i]; }
};

struct SearchResults {
  UninitBuffer<float> scores;            // nq * k (row-major; may be padded)
  UninitBuffer<std::uint64_t> ids;       // nq * k
  std::size_t k = 0;                     // stride / max candidates per query
  std::size_t nq = 0;
  /// If non-empty, per-query candidate counts (≤ k). Empty ⇒ every query has exactly k hits.
  std::vector<std::size_t> counts;
};

/// Optional wall-clock breakdown for one search call (milliseconds).
struct SearchStageTimings {
  /// Blocked FastScan / permute-dot over codes (`score_prepared`) — turbovec-comparable kernel.
  double fastscan_ms = 0.0;
};

struct PreparedQueryLut {
  std::vector<std::uint8_t> uint8_luts;
  float scale = 1.f;
  float bias = 0.f;
};

enum class SearchBackendKind { VmPermuteDot, VmVnni, Avx2, Scalar };

/// Query-side state shared across one or more blocked code ranges (buckets).
struct PreparedQueries {
  std::size_t nq = 0;
  std::size_t dim = 0;
  std::size_t bits = 0;
  std::size_t n_byte_groups = 0;
  SearchBackendKind backend = SearchBackendKind::Scalar;

  std::vector<float> q_rot;
  std::vector<float> bias_corrs;

  std::vector<QueryPermuteDot> pds;
  std::vector<std::vector<std::uint8_t>> split_luts;
  std::vector<float> lut_scales;
  std::vector<float> lut_biases;
  std::vector<PreparedQueryLut> luts;
};

PreparedQueries prepare_queries(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                const Rotation& rotation, std::span<const float> centroids,
                                std::size_t bits, std::span<const float> tqplus_shift,
                                std::span<const float> tqplus_scale);

/// Build LUTs from already-transformed query rows (no rotation). Used for landmark FastScan.
PreparedQueries prepare_from_rows(std::span<const float> rows, std::size_t nq, std::size_t dim,
                                  std::span<const float> centroids, std::size_t bits);

/// Sign-plane LUTs for staged 4-bit search (turbovec `build_sign_lut`).
/// Scanned as a 2-bit-shaped buffer with `n_byte_groups = dim/8`.
PreparedQueries prepare_sign_from_rows(std::span<const float> rows, std::size_t nq, std::size_t dim);

/// FWHT → 4-bit landmark project → FastScan top-k with residual-energy bound.
/// Score = s_L + ‖q_⊥‖·‖r_i‖. No full-dim re-rank.
/// If `timings` is non-null, fills `fastscan_ms` for the code-scan kernel only.
/// When `sign_codes` is non-empty (planes layout), runs sign shortlist → densified exact.
/// `codes_flat` is row-major `n × (scan_dim/(8/scan_bits))` for gathering the shortlist.
SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, const QuantizedLandmarks& landmarks,
                                     std::size_t scan_dim, std::span<const float> lm_centroids,
                                     std::size_t scan_bits,
                                     std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                                     std::span<const float> lm_scales,
                                     std::span<const float> residual_norms, const Rotation& rotation,
                                     std::span<const float> scale_resid = {},
                                     SearchStageTimings* timings = nullptr,
                                     std::span<const std::uint8_t> sign_codes = {},
                                     std::size_t sign_n_blocks = 0,
                                     std::span<const std::uint8_t> codes_flat = {});

/// Copy prepared state for a single query index (for per-query IVF probing).
PreparedQueries prepared_query_at(const PreparedQueries& prep, std::size_t qi);

/// Score a contiguous blocked range using prepared query state (top-k heap).
/// If `id_map` is non-empty it must have `scales.size()` entries; local ids are remapped.
/// If `query_indices` is non-empty, only those prepared queries are scored (output nq =
/// query_indices.size()); otherwise all `prep.nq` queries are scored.
/// Optional residual bound: score += residual_scales[qi] * residual_norms[vi].
/// If `hit_bitmaps` is non-null it must have length `prep.nq` (or `query_indices.size()`
/// when that is non-empty); each entry masks which vectors may enter that query's heap.
/// Empty blocks are skipped inside the existing permute-dot / VNNI kernels.
/// If `buffered`, uses turbovec H99 collector (capacity `k`, returns top `k/2`).
/// Optional `seed_mins` (length `prep.nq`) seeds each query's admit threshold.
SearchResults score_prepared(const PreparedQueries& prep, std::size_t k,
                             std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                             std::span<const float> scales,
                             std::span<const std::uint64_t> id_map = {},
                             std::span<const std::size_t> query_indices = {},
                             std::span<const float> residual_norms = {},
                             std::span<const float> residual_scales = {},
                             std::span<const float> scale_resid = {},
                             const std::uint64_t* const* hit_bitmaps = nullptr,
                             bool buffered = false, const float* seed_mins = nullptr);

/// Continue scoring into existing per-query top-k heaps (length `prep.nq`, indexed by global qi).
/// Only `query_indices` are updated. Heaps may already be partially filled.
/// When `id_map` is non-empty, pushes `id_map[local]` (required for multi-bucket IVF).
/// If `score_offsets` is non-null (length `prep.nq`), adds `score_offsets[qi]` after α·LUT
/// (IVF residual: ⟨q̂, c⟩ + α · ⟨q̂, r̂⟩) without rebuilding query LUTs.
void score_prepared_into(const PreparedQueries& prep, std::size_t k,
                         std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                         std::span<const float> scales, std::span<const std::uint64_t> id_map,
                         std::span<const std::size_t> query_indices, float* heap_s,
                         std::uint64_t* heap_i, std::size_t* heap_sz, float* heap_min,
                         std::size_t* heap_mi, const float* score_offsets = nullptr);

/// Sort unordered top-k heaps (min-heap layout) into descending SearchResults.
void heaps_to_search_results(SearchResults& out, std::size_t nq, std::size_t k,
                             const float* heap_s, const std::uint64_t* heap_i,
                             const std::size_t* heap_sz);

SearchResults search_flat(std::span<const float> queries, std::size_t nq, std::size_t dim,
                          std::size_t k, const Rotation& rotation,
                          std::span<const float> centroids, std::size_t bits,
                          std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                          std::span<const float> scales, std::span<const float> tqplus_shift,
                          std::span<const float> tqplus_scale);

/// Merge `src` into `dst` keeping top-k per query (higher score wins; tie → lower id).
void merge_search_results(SearchResults& dst, const SearchResults& src);

}  // namespace vectorcache
