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
#include "vectorcache/query/query_config.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

#if defined(VECTORCACHE_QUERY_OPENMP) && VECTORCACHE_QUERY_OPENMP
#include <omp.h>
#endif

namespace vectorcache::query {

namespace {

#if VECTORCACHE_QUERY_DEPTH >= 3
inline void prefetch_l0(const ingest::LogicalBlock& block, std::size_t i) {
  if (i < block.len()) {
    _mm_prefetch(reinterpret_cast<const char*>(block.vector_l0(i).data()), _MM_HINT_T0);
  }
}

inline void prefetch_full(const ingest::LogicalBlock& block, std::size_t i) {
  if (i < block.len()) {
    _mm_prefetch(reinterpret_cast<const char*>(block.vector_full(i).data()), _MM_HINT_T0);
  }
}
#endif

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

void search_block_depth1(const ingest::LogicalBlock& block, std::size_t base_id,
                         const PreparedQuery& query, float l1_block_threshold, TopKHits& topk) {
  const std::size_t l1_bits = quantize::l1_bits_per_vector(block.layout().padded_dim);
  const std::size_t l1_words = block.layout().l1_words_per_vec;
  const auto l1_slice = block.l1_slice();
  const std::size_t n = block.len();

  std::vector<float> l1_scores(n);
  bit_agreement_batch(query.l1, l1_bits, l1_slice, l1_words, n, l1_scores);

  float block_best = -1.0f;
  for (std::size_t i = 0; i < n; ++i) {
    block_best = std::max(block_best, l1_scores[i]);
  }
  if (block_best < l1_block_threshold) {
    return;
  }

  for (std::size_t i = 0; i < n; ++i) {
    topk.push(base_id + i, l1_scores[i]);
  }
}

#if VECTORCACHE_QUERY_DEPTH >= 2

void search_block_depth2(const ingest::LogicalBlock& block, std::size_t base_id,
                         const PreparedQuery& query, float l1_block_threshold,
                         float l0_vector_threshold, TopKHits& topk) {
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
  if (block_best < l1_block_threshold) {
    return;
  }

  const ingest::BlockLayout& layout = block.layout();
  for (std::size_t sub = 0; sub < layout.l0_phys_blocks; ++sub) {
    const std::size_t sub_start = sub * layout.l0_vecs_per_sub;
    if (sub_start >= n) {
      break;
    }
    const std::size_t sub_len = std::min(layout.l0_vecs_per_sub, n - sub_start);
    const auto* l0_base = reinterpret_cast<const std::uint64_t*>(
        block.bytes().data() + layout.l0_sub_offset(sub));
    std::vector<float> l0_scores(sub_len);
    bit_agreement_batch(query.l0, l0_bits,
                        {l0_base, sub_len * layout.l0_words_per_vec}, layout.l0_words_per_vec,
                        sub_len, l0_scores);
    for (std::size_t j = 0; j < sub_len; ++j) {
      if (l0_scores[j] >= l0_vector_threshold) {
        topk.push(base_id + sub_start + j, l0_scores[j]);
      }
    }
  }
}

#endif

#if VECTORCACHE_QUERY_DEPTH >= 3

void search_block_depth3(const ingest::LogicalBlock& block, std::size_t base_id,
                         const PreparedQuery& query, const QueryParams& params, TopKHits& topk) {
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

  struct Candidate {
    std::size_t local_idx;
    float l0_score;
  };
  std::vector<Candidate> survivors;
  survivors.reserve(n);

  const ingest::BlockLayout& layout = block.layout();
  for (std::size_t sub = 0; sub < layout.l0_phys_blocks; ++sub) {
    const std::size_t sub_start = sub * layout.l0_vecs_per_sub;
    if (sub_start >= n) {
      break;
    }
    const std::size_t sub_len = std::min(layout.l0_vecs_per_sub, n - sub_start);
    const auto* l0_base = reinterpret_cast<const std::uint64_t*>(
        block.bytes().data() + layout.l0_sub_offset(sub));
    std::vector<float> l0_scores(sub_len);
    bit_agreement_batch(query.l0, l0_bits,
                        {l0_base, sub_len * layout.l0_words_per_vec}, layout.l0_words_per_vec,
                        sub_len, l0_scores);
    for (std::size_t j = 0; j < sub_len; ++j) {
      if (l0_scores[j] >= params.l0_vector_threshold) {
        survivors.push_back({sub_start + j, l0_scores[j]});
      }
    }
  }

  if (params.l0_dot_promote > 0 && survivors.size() > params.l0_dot_promote) {
    std::nth_element(survivors.begin(), survivors.begin() + params.l0_dot_promote, survivors.end(),
                     [](const Candidate& a, const Candidate& b) { return a.l0_score > b.l0_score; });
    survivors.resize(params.l0_dot_promote);
  }

  for (std::size_t s = 0; s < survivors.size(); ++s) {
    const std::size_t i = survivors[s].local_idx;
    if (s + 2 < survivors.size()) {
      prefetch_l0(block, survivors[s + 2].local_idx);
      prefetch_full(block, survivors[s + 2].local_idx);
    }
    const float score = dot_f32(query.rotated, block.vector_full(i));
    topk.push(base_id + i, score);
  }
}

#endif

void search_block(const BlockRef& ref, const PreparedQuery& query, const QueryParams& params,
                  TopKHits& topk) {
  if (ref.block == nullptr || ref.block->is_empty()) {
    return;
  }
#if VECTORCACHE_QUERY_DEPTH == 1
  search_block_depth1(*ref.block, ref.base_id, query, params.l1_block_threshold, topk);
#elif VECTORCACHE_QUERY_DEPTH == 2
  search_block_depth2(*ref.block, ref.base_id, query, params.l1_block_threshold,
                      params.l0_vector_threshold, topk);
#else
  search_block_depth3(*ref.block, ref.base_id, query, params, topk);
#endif
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
#if VECTORCACHE_QUERY_DEPTH >= 2
  const std::size_t l0_words = store.l0_words_per_vec();
#endif

  PreparedQuery prepared;
  prepared.rotated.assign(padded_dim, 0.0f);
  prepared.l1.assign(l1_words, 0);
#if VECTORCACHE_QUERY_DEPTH >= 2
  prepared.l0.assign(l0_words, 0);
#endif

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
#if VECTORCACHE_QUERY_DEPTH >= 2
  quantize::quantize_1dim_to_1bit_into(prepared.rotated, prepared.l0);
#endif

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
