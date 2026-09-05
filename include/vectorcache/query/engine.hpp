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
  std::size_t k = 10;
};

struct PreparedQuery {
  AlignedVector<float> rotated;
  std::uint8_t parent_key = 0;
  AlignedVector<std::uint64_t> l0;
};

class QueryEngine {
 public:
  static QueryEngine with_rotation(const ingest::ParentStore& store, std::size_t input_dim,
                                   std::uint64_t seed);
  static QueryEngine from_rotated(const ingest::ParentStore& store);

  PreparedQuery prepare(std::span<const float> query) const;
  /// Reuse rotated/L0 buffers in out across calls (resizes only when dimensions change).
  void prepare_into(PreparedQuery& out, std::span<const float> query) const;
  std::vector<QueryHit> search(std::span<const float> query, const QueryParams& params) const;
  std::vector<QueryHit> search_prepared(const PreparedQuery& prepared,
                                        const QueryParams& params) const;

 private:
  QueryEngine(const ingest::ParentStore& store, std::optional<transform::SrhtRotation> rotation,
              bool query_is_rotated, std::size_t input_dim);

  const ingest::ParentStore& store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool query_is_rotated_;
  std::size_t input_dim_;
};

}  // namespace vectorcache::query
