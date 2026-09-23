#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectorcache {

/// Compute `scores[i * n_centroids + j] = ⟨rows[i], centroids[j]⟩` for unit (or arbitrary) rows.
/// `rows` is flat `n_rows * dim`, `centroids` is flat `n_centroids * dim`.
void score_against_centroids(std::span<const float> rows, std::size_t n_rows, std::size_t dim,
                             std::span<const float> centroids, std::size_t n_centroids,
                             std::span<float> scores_out);

/// For each row, write the centroid index with maximum dot product into `assign_out[i]`.
void assign_nearest_centroid(std::span<const float> rows, std::size_t n_rows, std::size_t dim,
                             std::span<const float> centroids, std::size_t n_centroids,
                             std::span<std::uint32_t> assign_out);

/// L2-normalize rows in place (8-way chain norm, matching encode).
void normalize_rows_inplace(std::span<float> rows, std::size_t n_rows, std::size_t dim);

}  // namespace vectorcache
