#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "vectorcache/search/search_vnni.hpp"
#include "vectorcache/transform/rotation.hpp"

namespace vectorcache {

struct SearchResults {
  std::vector<float> scores;       // nq * k
  std::vector<std::uint64_t> ids;  // nq * k
  std::size_t k = 0;
  std::size_t nq = 0;
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

/// Copy prepared state for a single query index (for per-query IVF probing).
PreparedQueries prepared_query_at(const PreparedQueries& prep, std::size_t qi);

/// Score a contiguous blocked range using prepared query state.
/// If `id_map` is non-empty it must have `scales.size()` entries; local ids are remapped.
/// If `query_indices` is non-empty, only those prepared queries are scored (output nq =
/// query_indices.size()); otherwise all `prep.nq` queries are scored.
SearchResults score_prepared(const PreparedQueries& prep, std::size_t k,
                             std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                             std::span<const float> scales,
                             std::span<const std::uint64_t> id_map = {},
                             std::span<const std::size_t> query_indices = {});

SearchResults search_flat(std::span<const float> queries, std::size_t nq, std::size_t dim,
                          std::size_t k, const Rotation& rotation,
                          std::span<const float> centroids, std::size_t bits,
                          std::span<const std::uint8_t> blocked_codes, std::size_t n_blocks,
                          std::span<const float> scales, std::span<const float> tqplus_shift,
                          std::span<const float> tqplus_scale);

/// Merge `src` into `dst` keeping top-k per query (higher score wins; tie → lower id).
void merge_search_results(SearchResults& dst, const SearchResults& src);

}  // namespace vectorcache
