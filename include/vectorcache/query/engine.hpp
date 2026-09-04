#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/transform/srht.hpp"

namespace vectorcache::query {

struct QueryHit {
  std::size_t id = 0;
  float score = 0.0f;
};

struct QueryParams {
  float l1_block_threshold = 0.0f;
  float l1_vector_threshold = 0.0f;
  float l0_vector_threshold = 0.0f;
  std::size_t k = 10;
  /// When > 0, only search the top N blocks by max L1 score (block routing).
  std::size_t top_blocks = 0;
};

struct PreparedQuery {
  AlignedVector<float> rotated;
  AlignedVector<std::uint64_t> l1;
  AlignedVector<std::uint64_t> l0;
};

class QueryEngine {
 public:
  static QueryEngine with_rotation(const ingest::BlockStore& store, std::size_t input_dim,
                                   std::uint64_t seed);
  static QueryEngine from_rotated(const ingest::BlockStore& store);

  PreparedQuery prepare(std::span<const float> query) const;
  std::vector<QueryHit> search(std::span<const float> query, const QueryParams& params) const;
  std::vector<QueryHit> search_prepared(const PreparedQuery& prepared,
                                          const QueryParams& params) const;
  std::vector<std::vector<QueryHit>> search_batch(std::span<const std::span<const float>> queries,
                                                  const QueryParams& params) const;

 private:
  QueryEngine(const ingest::BlockStore& store, std::optional<transform::SrhtRotation> rotation,
              bool query_is_rotated, std::size_t input_dim);

  PreparedQuery prepare_query_impl(std::span<const float> query) const;

  const ingest::BlockStore& store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool query_is_rotated_;
  std::size_t input_dim_;
};

}  // namespace vectorcache::query
