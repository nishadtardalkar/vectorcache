#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "vectorcache/search/search_vnni.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

/// Contiguous buffer that can grow without value-initializing (threshold shortlists).
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

/// Wall-clock stage times for one `search_landmark_rerank` call (total batch, not per-query).
struct LandmarkStageTimings {
  // Coarse (sums of the detail fields below where applicable).
  double project_ms = 0;         // rotate + landmark_project
  double shortlist_prep_ms = 0;  // landmark LUT / permute-dot build
  double shortlist_scan_ms = 0;  // buf_alloc + FastScan kernel
  double rerank_prep_ms = 0;     // full-dim LUT build
  double rerank_ms = 0;          // score + select

  // Fine-grained kernel steps.
  double rotate_ms = 0;            // L2 + FWHT
  double landmark_project_ms = 0;  // q_rot · L
  double shortlist_buf_ms = 0;     // allocate/zero shortlist score+id buffers
  double shortlist_kernel_ms = 0;  // permute-dot / threshold fill
  double shortlist_copy_ms = 0;    // unused (bitmap path); kept for timing ABI
  double rerank_score_ms = 0;      // pass-2 masked full-dim LUT over bitmap hits
  double rerank_select_ms = 0;     // top-k heap + sort

  double mean_shortlist = 0;  // mean candidates / query after shortlist stage
  double max_shortlist = 0;   // max candidates on any query
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

/// FWHT → landmark project → FastScan threshold shortlist → quantized full-dim re-rank.
/// Requires finite `score_threshold`. Keeps landmark scores ≥ threshold into a fixed buffer of
/// size `2×shortlist` (stop when full; `shortlist` is the mean target; 0 = uncapped → n_vectors).
/// If `timings` is non-null, fills per-stage wall times for this call.
SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, std::size_t shortlist,
                                     std::span<const float> landmark_basis, std::size_t n_landmarks,
                                     std::size_t scan_dim, std::span<const float> lm_centroids,
                                     std::size_t scan_bits, std::span<const std::uint8_t> blocked_codes,
                                     std::size_t n_blocks, std::span<const float> lm_scales,
                                     std::span<const std::uint8_t> rerank_codes,
                                     std::span<const std::uint8_t> rerank_blocked,
                                     std::size_t rerank_n_blocks, std::span<const float> rerank_scales,
                                     std::span<const float> centroids, std::size_t rerank_bits,
                                     const Rotation& rotation, float score_threshold,
                                     LandmarkStageTimings* timings = nullptr,
                                     bool shortlist_only = false);

/// Copy prepared state for a single query index (for per-query IVF probing).
PreparedQueries prepared_query_at(const PreparedQueries& prep, std::size_t qi);

/// Score a contiguous blocked range using prepared query state.
/// If `id_map` is non-empty it must have `scales.size()` entries; local ids are remapped.
/// If `query_indices` is non-empty, only those prepared queries are scored (output nq =
/// query_indices.size()); otherwise all `prep.nq` queries are scored.
/// Finite `score_threshold` ⇒ fixed-buffer collect (no top-k heap).
SearchResults score_prepared(const PreparedQueries& prep, std::size_t k,
                             std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                             std::span<const float> scales,
                             std::span<const std::uint64_t> id_map = {},
                             std::span<const std::size_t> query_indices = {},
                             float score_threshold = std::numeric_limits<float>::quiet_NaN(),
                             LandmarkStageTimings* timings = nullptr);

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
