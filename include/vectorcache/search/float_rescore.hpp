#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/search/search.hpp"

namespace vectorcache {

/// Exact IP re-rank of an ANN top-k using unit-normalized float rows.
/// `queries` and `corpus` are row-major L2-normalized (same space as recall GT).
/// `corpus.size()` must be `n_vectors * dim` with `n_vectors >= max(id)+1`.
/// Rewrites `out.scores` / reorders `out.ids` descending; intended to run inside
/// the timed search path. AVX2+FMA when available, scalar fallback otherwise.
void float_rescore_topk(SearchResults& out, std::span<const float> queries,
                        std::span<const float> corpus, std::size_t dim);

}  // namespace vectorcache
