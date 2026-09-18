#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/query/fastscan.hpp"
#include "vectorcache/transform/srht.hpp"

namespace vectorcache::query {

struct QueryHit {
  std::size_t id = 0;
  float score = 0.0f;
};

struct QueryParams {
  std::size_t k = 10;
  /// Multi-probe L_∞ radius in bin units (cells with max |o_i| <= probe_radius).
  std::size_t probe_radius = 1;
};

struct SearchStats {
  std::size_t candidates = 0;
  std::size_t cells_probed = 0;
};

struct PreparedQuery {
  AlignedVector<float> rotated;
};

class QueryEngine {
 public:
  static QueryEngine with_rotation(const ingest::VectorStore& store, std::size_t input_dim,
                                   std::uint64_t seed);
  static QueryEngine from_rotated(const ingest::VectorStore& store);

  PreparedQuery prepare(std::span<const float> query) const;
  void prepare_into(PreparedQuery& out, std::span<const float> query) const;
  std::vector<QueryHit> search(std::span<const float> query, const QueryParams& params,
                               SearchStats* stats = nullptr) const;
  std::vector<QueryHit> search_prepared(const PreparedQuery& prepared, const QueryParams& params,
                                        SearchStats* stats = nullptr) const;

  const quantize::LloydMaxCodebook& codebook() const { return codebook_; }

  /// Ensure FastScan blocked cache matches current store size (also done lazily on search).
  void prepare_index() const;

 private:
  QueryEngine(const ingest::VectorStore& store, std::optional<transform::SrhtRotation> rotation,
              bool query_is_rotated, std::size_t input_dim, quantize::LloydMaxCodebook codebook);

  const BlockedCodes& blocked_codes() const;

  const ingest::VectorStore& store_;
  std::optional<transform::SrhtRotation> rotation_;
  bool query_is_rotated_;
  std::size_t input_dim_;
  quantize::LloydMaxCodebook codebook_;
  mutable BlockedCodes blocked_;
  mutable std::size_t blocked_n_ = 0;
};

}  // namespace vectorcache::query
