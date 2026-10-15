#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace vectorcache {

struct SearchResults {
  std::vector<float> scores;       // nq * k
  std::vector<std::uint64_t> ids;  // nq * k
  std::size_t k = 0;
  std::size_t nq = 0;
};

/// Blocked float landmark shortlist scan + exact float re-rank.
SearchResults search_landmark_rerank(std::span<const float> queries, std::size_t nq, std::size_t dim,
                                     std::size_t k, std::size_t shortlist,
                                     std::span<const float> landmark_basis,  // dim * m, col-major
                                     std::size_t n_landmarks,
                                     std::span<const float> blocked_landmarks,  // n_blocks*m*kBlock
                                     std::size_t n_vectors, std::size_t n_blocks,
                                     std::span<const float> db_unit,  // n_vectors * dim
                                     const class Rotation& rotation);

std::string search_backend_name(std::size_t /*bits*/, std::size_t /*dim*/);
std::string search_backend_diagnostics(std::size_t /*bits*/, std::size_t /*dim*/);

}  // namespace vectorcache
