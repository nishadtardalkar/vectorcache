#include "vectorcache/query/engine.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <queue>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/block.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

#if defined(VECTORCACHE_QUERY_OPENMP) && VECTORCACHE_QUERY_OPENMP
#include <omp.h>
#endif

namespace vectorcache::query {

namespace {

inline void prefetch_l0(const ingest::LogicalBlock& block, std::size_t i) {
  if (i < block.len()) {
    _mm_prefetch(reinterpret_cast<const char*>(block.vector_l0(i).data()), _MM_HINT_T0);
  }
}

class TopKHits {
 public:
  explicit TopKHits(std::size_t k) : k_(k) {}

  void push(std::size_t id, float score) {
    if (k_ == 0) {
      return;
    }
    if (heap_.size() < k_) {
      heap_.push({id, score});
      return;
    }
    if (score > heap_.top().score) {
      heap_.pop();
      heap_.push({id, score});
    }
  }

  void merge_from(const TopKHits& other) {
    auto copy = other.heap_;
    while (!copy.empty()) {
      const QueryHit hit = copy.top();
      copy.pop();
      push(hit.id, hit.score);
    }
  }

  std::vector<QueryHit> finalize() const {
    std::vector<QueryHit> out;
    out.reserve(heap_.size());
    auto copy = heap_;
    while (!copy.empty()) {
      out.push_back(copy.top());
      copy.pop();
    }
    std::sort(out.begin(), out.end(),
              [](const QueryHit& a, const QueryHit& b) { return a.score > b.score; });
    return out;
  }

 private:
  struct HitGreater {
    bool operator()(const QueryHit& a, const QueryHit& b) const { return a.score > b.score; }
  };

  std::size_t k_;
  std::priority_queue<QueryHit, std::vector<QueryHit>, HitGreater> heap_;
};

struct BlockRef {
  const ingest::LogicalBlock* block = nullptr;
  std::size_t base_id = 0;
  float max_l1 = -1.0f;
};

void search_block(const BlockRef& ref, const PreparedQuery& query, const QueryParams& params,
                  TopKHits& topk) {
  if (ref.block == nullptr || ref.block->is_empty()) {
    return;
  }

  const ingest::LogicalBlock& block = *ref.block;
  const std::size_t l1_bits = quantize::l1_bits_per_vector(block.layout().padded_dim);
  const std::size_t l0_bits = quantize::l0_bits_per_vector(block.layout().padded_dim);
  const std::size_t n = block.len();

  std::vector<float> l1_scores(n);
  bit_agreement_batch(query.l1, l1_bits, block.l1_slice(), block.layout().l1_words_per_vec, n,
                      l1_scores);

  float block_best = -1.0f;
  for (std::size_t i = 0; i < n; ++i) {
    block_best = std::max(block_best, l1_scores[i]);
  }
  if (block_best < params.l1_block_threshold) {
    return;
  }

  std::vector<std::size_t> survivors;
  survivors.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    if (l1_scores[i] >= params.l1_vector_threshold) {
      survivors.push_back(i);
    }
  }

  for (std::size_t s = 0; s < survivors.size(); ++s) {
    if (s + 2 < survivors.size()) {
      prefetch_l0(block, survivors[s + 2]);
    }
    const std::size_t i = survivors[s];
    const float l0_score = bit_agreement_score(query.l0, block.vector_l0(i), l0_bits);
    if (l0_score >= params.l0_vector_threshold) {
      topk.push(ref.base_id + i, l0_score);
    }
  }
}

std::vector<BlockRef> build_block_refs(const ingest::BlockStore& store) {
  std::vector<BlockRef> refs;
  refs.reserve(store.block_count() + 1);
  std::size_t base_id = 0;
  for (const auto& block : store.blocks()) {
    refs.push_back({&block, base_id, -1.0f});
    base_id += block.len();
  }
  refs.push_back({&store.partial_block(), base_id, -1.0f});
  return refs;
}

void score_blocks_l1(const PreparedQuery& query, const ingest::BlockStore& store,
                     std::vector<BlockRef>& refs) {
  const std::size_t l1_bits = quantize::l1_bits_per_vector(store.padded_dim());
  for (auto& ref : refs) {
    if (ref.block == nullptr || ref.block->is_empty()) {
      ref.max_l1 = -1.0f;
      continue;
    }
    const std::size_t n = ref.block->len();
    std::vector<float> l1_scores(n);
    bit_agreement_batch(query.l1, l1_bits, ref.block->l1_slice(), store.l1_words_per_vec(), n,
                        l1_scores);
    ref.max_l1 = -1.0f;
    for (float s : l1_scores) {
      ref.max_l1 = std::max(ref.max_l1, s);
    }
  }
}

void search_store(const ingest::BlockStore& store, const PreparedQuery& query,
                  const QueryParams& params, TopKHits& topk) {
  std::vector<BlockRef> refs = build_block_refs(store);

  if (params.top_blocks > 0) {
    score_blocks_l1(query, store, refs);
    std::sort(refs.begin(), refs.end(),
              [](const BlockRef& a, const BlockRef& b) { return a.max_l1 > b.max_l1; });
    if (refs.size() > params.top_blocks) {
      refs.resize(params.top_blocks);
    }
  }

#if defined(VECTORCACHE_QUERY_OPENMP) && VECTORCACHE_QUERY_OPENMP
  const int nblocks = static_cast<int>(refs.size());
  std::vector<TopKHits> local(static_cast<std::size_t>(nblocks), TopKHits(params.k));
#pragma omp parallel for schedule(static)
  for (int b = 0; b < nblocks; ++b) {
    search_block(refs[static_cast<std::size_t>(b)], query, params,
                 local[static_cast<std::size_t>(b)]);
  }
  for (const auto& part : local) {
    topk.merge_from(part);
  }
#else
  for (const auto& ref : refs) {
    search_block(ref, query, params, topk);
  }
#endif
}

PreparedQuery make_prepared_query(const ingest::BlockStore& store,
                                  const std::optional<transform::SrhtRotation>& rotation,
                                  bool query_is_rotated, std::size_t input_dim,
                                  std::span<const float> query) {
  const std::size_t padded_dim = store.padded_dim();
  const std::size_t l1_words = store.l1_words_per_vec();
  const std::size_t l0_words = store.l0_words_per_vec();

  PreparedQuery prepared;
  prepared.rotated.assign(padded_dim, 0.0f);
  prepared.l1.assign(l1_words, 0);
  prepared.l0.assign(l0_words, 0);

  if (query_is_rotated) {
    if (query.size() != padded_dim) {
      throw Error("rotated query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), padded_dim * sizeof(float));
  } else if (rotation.has_value()) {
    if (query.size() != input_dim) {
      throw Error("query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), input_dim * sizeof(float));
    if (padded_dim > input_dim) {
      std::memset(prepared.rotated.data() + input_dim, 0, (padded_dim - input_dim) * sizeof(float));
    }
    transform::l2_normalize_in_place(prepared.rotated);
    rotation->apply_in_place(prepared.rotated);
  } else {
    throw Error("QueryEngine requires with_rotation() or from_rotated()");
  }

  quantize::quantize_4d_to_1bit_into(prepared.rotated, prepared.l1);
  quantize::quantize_1dim_to_1bit_into(prepared.rotated, prepared.l0);

  return prepared;
}

}  // namespace

QueryEngine::QueryEngine(const ingest::BlockStore& store,
                         std::optional<transform::SrhtRotation> rotation, bool query_is_rotated,
                         std::size_t input_dim)
    : store_(store),
      rotation_(std::move(rotation)),
      query_is_rotated_(query_is_rotated),
      input_dim_(input_dim) {}

QueryEngine QueryEngine::with_rotation(const ingest::BlockStore& store, std::size_t input_dim,
                                       std::uint64_t seed) {
  return QueryEngine(store, transform::SrhtRotation(input_dim, seed), false, input_dim);
}

QueryEngine QueryEngine::from_rotated(const ingest::BlockStore& store) {
  return QueryEngine(store, std::nullopt, true, store.padded_dim());
}

PreparedQuery QueryEngine::prepare_query_impl(std::span<const float> query) const {
  return make_prepared_query(store_, rotation_, query_is_rotated_, input_dim_, query);
}

PreparedQuery QueryEngine::prepare(std::span<const float> query) const {
  return prepare_query_impl(query);
}

std::vector<QueryHit> QueryEngine::search_prepared(const PreparedQuery& prepared,
                                                   const QueryParams& params) const {
  TopKHits topk(params.k);
  search_store(store_, prepared, params, topk);
  return topk.finalize();
}

std::vector<QueryHit> QueryEngine::search(std::span<const float> query,
                                          const QueryParams& params) const {
  return search_prepared(prepare(query), params);
}

std::vector<std::vector<QueryHit>> QueryEngine::search_batch(
    std::span<const std::span<const float>> queries, const QueryParams& params) const {
  std::vector<std::vector<QueryHit>> results;
  results.reserve(queries.size());
  for (const auto query : queries) {
    results.push_back(search(query, params));
  }
  return results;
}

}  // namespace vectorcache::query
