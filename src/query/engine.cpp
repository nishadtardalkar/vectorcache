#include "vectorcache/query/engine.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/index/rp_buckets.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/query/fastscan.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/simd.hpp"
#include "vectorcache/transform/normalize.hpp"

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
#include <omp.h>
#endif

namespace vectorcache::query {

namespace {

constexpr std::size_t kHeapTopKThreshold = 32;
/// Parallelize when a single cell covers at least this many FastScan blocks.
constexpr std::size_t kParallelMinBlocks = 1024;

class TopKHits {
 public:
  explicit TopKHits(std::size_t k)
      : k_(k),
        use_heap_(k > kHeapTopKThreshold),
        size_(0),
        min_score_(0.0f),
        min_idx_(0),
        scores_(use_heap_ ? 0 : k, 0.0f),
        ids_(use_heap_ ? 0 : k, 0) {
    if (use_heap_) {
      heap_.reserve(k);
    }
  }

  void clear() {
    size_ = 0;
    min_score_ = 0.0f;
    min_idx_ = 0;
    heap_.clear();
  }

  float reject_threshold() const {
    if (k_ == 0) {
      return std::numeric_limits<float>::infinity();
    }
    if (use_heap_) {
      if (heap_.size() < k_) {
        return -std::numeric_limits<float>::infinity();
      }
      return heap_.front().score;
    }
    if (size_ < k_) {
      return -std::numeric_limits<float>::infinity();
    }
    return min_score_;
  }

  void push(std::size_t id, float score) {
    if (k_ == 0) {
      return;
    }
    if (use_heap_) {
      push_heap(id, score);
      return;
    }
    if (size_ < k_) {
      scores_[size_] = score;
      ids_[size_] = id;
      ++size_;
      if (size_ == k_) {
        rescan_min();
      }
      return;
    }
    if (score > min_score_ || (score == min_score_ && id < ids_[min_idx_])) {
      scores_[min_idx_] = score;
      ids_[min_idx_] = id;
      rescan_min();
    }
  }

  void merge_from(const TopKHits& other) {
    if (other.use_heap_) {
      for (const auto& hit : other.heap_) {
        push(hit.id, hit.score);
      }
      return;
    }
    for (std::size_t i = 0; i < other.size_; ++i) {
      push(other.ids_[i], other.scores_[i]);
    }
  }

  std::vector<QueryHit> finalize() const {
    std::vector<QueryHit> out;
    if (use_heap_) {
      std::vector<HeapHit> sorted = heap_;
      if (!sorted.empty()) {
        std::sort_heap(sorted.begin(), sorted.end(), heap_worse);
        std::reverse(sorted.begin(), sorted.end());
        // sort_heap+reverse flips id order within equal-score runs; restore ascending id.
        for (std::size_t i = 0; i < sorted.size();) {
          std::size_t j = i + 1;
          while (j < sorted.size() && sorted[j].score == sorted[i].score) {
            ++j;
          }
          std::reverse(sorted.begin() + static_cast<std::ptrdiff_t>(i),
                       sorted.begin() + static_cast<std::ptrdiff_t>(j));
          i = j;
        }
      }
      out.reserve(sorted.size());
      for (const auto& hit : sorted) {
        out.push_back({hit.id, hit.score});
      }
      return out;
    }
    out.reserve(size_);
    for (std::size_t i = 0; i < size_; ++i) {
      out.push_back({ids_[i], scores_[i]});
    }
    std::sort(out.begin(), out.end(), [](const QueryHit& a, const QueryHit& b) {
      if (a.score != b.score) {
        return a.score > b.score;
      }
      return a.id < b.id;
    });
    return out;
  }

 private:
  struct HeapHit {
    float score;
    std::size_t id;
  };

  static bool heap_worse(const HeapHit& a, const HeapHit& b) {
    if (a.score != b.score) {
      return a.score > b.score;
    }
    return a.id < b.id;
  }

  void push_heap(std::size_t id, float score) {
    if (heap_.size() < k_) {
      heap_.push_back({score, id});
      std::push_heap(heap_.begin(), heap_.end(), heap_worse);
      return;
    }
    const auto& worst = heap_.front();
    if (score > worst.score || (score == worst.score && id < worst.id)) {
      std::pop_heap(heap_.begin(), heap_.end(), heap_worse);
      heap_.back() = {score, id};
      std::push_heap(heap_.begin(), heap_.end(), heap_worse);
    }
  }

  void rescan_min() {
    std::size_t mi = 0;
    for (std::size_t h = 1; h < size_; ++h) {
      if (scores_[h] < scores_[mi] || (scores_[h] == scores_[mi] && ids_[h] > ids_[mi])) {
        mi = h;
      }
    }
    min_score_ = scores_[mi];
    min_idx_ = mi;
  }

  std::size_t k_;
  bool use_heap_;
  std::size_t size_;
  float min_score_;
  std::size_t min_idx_;
  std::vector<float> scores_;
  std::vector<std::size_t> ids_;
  std::vector<HeapHit> heap_;
};

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
std::vector<TopKHits>& omp_topk_scratch(std::size_t k, int nthreads) {
  static std::vector<TopKHits> locals;
  static std::size_t locals_k = 0;
  if (locals.size() != static_cast<std::size_t>(nthreads) || locals_k != k) {
    locals.clear();
    locals.reserve(static_cast<std::size_t>(nthreads));
    for (int t = 0; t < nthreads; ++t) {
      locals.emplace_back(k);
    }
    locals_k = k;
  } else {
    for (auto& local : locals) {
      local.clear();
    }
  }
  return locals;
}
#endif

void apply_scales(std::span<float> scores, std::span<const float> scales, std::size_t base,
                  std::size_t count) {
  std::size_t i = 0;
  for (; i + simd::kWidth <= count; i += simd::kWidth) {
    const __m512 s = _mm512_loadu_ps(scores.data() + i);
    const __m512 a = _mm512_loadu_ps(scales.data() + base + i);
    _mm512_storeu_ps(scores.data() + i, _mm512_mul_ps(s, a));
  }
  for (; i < count; ++i) {
    scores[i] *= scales[base + i];
  }
}

void score_block_range_into_topk(const BlockedCodes& blocked, const QueryLut& lut,
                                 const ingest::VectorStore& store, std::size_t block,
                                 std::size_t range_lo, std::size_t range_hi, TopKHits& topk) {
  const std::size_t count = blocked.block_count(block);
  if (count == 0) {
    return;
  }
  const std::size_t base = block * BlockedCodes::kBlock;
  const std::size_t block_hi = base + count;
  if (block_hi <= range_lo || base >= range_hi) {
    return;
  }

  alignas(64) float score_buf[BlockedCodes::kBlock];
  score_blocked_batch(lut, blocked, block, std::span<float>(score_buf, count));

  const auto ids = store.ids();
  const auto scales = store.scales();
  apply_scales(std::span<float>(score_buf, count), scales, base, count);

  const std::size_t begin = std::max(range_lo, base);
  const std::size_t end = std::min(range_hi, block_hi);
  float threshold = topk.reject_threshold();
  for (std::size_t idx = begin; idx < end; ++idx) {
    const float score = score_buf[idx - base];
    if (score < threshold) [[likely]] {
      continue;
    }
    topk.push(ids[idx], score);
    threshold = topk.reject_threshold();
  }
}

void score_store_range_fastscan(const BlockedCodes& blocked, const QueryLut& lut,
                                const ingest::VectorStore& store, std::size_t lo, std::size_t hi,
                                TopKHits& topk) {
  if (lo >= hi) {
    return;
  }
  const std::size_t block_begin = lo / BlockedCodes::kBlock;
  const std::size_t block_end = (hi + BlockedCodes::kBlock - 1) / BlockedCodes::kBlock;
  for (std::size_t block = block_begin; block < block_end; ++block) {
    score_block_range_into_topk(blocked, lut, store, block, lo, hi, topk);
  }
}

void score_store_range_vector_major(const ingest::VectorStore& store, const QueryLut& lut,
                                    const PreparedQuery& query,
                                    const quantize::LloydMaxCodebook& codebook, std::size_t lo,
                                    std::size_t hi, TopKHits& topk) {
  if (lo >= hi) {
    return;
  }
  constexpr std::size_t kScoreChunk = 512;
  const std::size_t words = store.l0_words_per_vec();
  const auto codes = store.l0_codes();
  const auto ids = store.ids();
  const auto scales = store.scales();

  alignas(64) float score_buf[kScoreChunk];
  float threshold = topk.reject_threshold();
  for (std::size_t base = lo; base < hi; base += kScoreChunk) {
    const std::size_t chunk = std::min(kScoreChunk, hi - base);
    asymmetric_ip_batch_lut(lut, codes.subspan(base * words, chunk * words), words, chunk, codebook,
                            query.rotated, std::span<float>(score_buf, chunk));
    apply_scales(std::span<float>(score_buf, chunk), scales, base, chunk);
    for (std::size_t i = 0; i < chunk; ++i) {
      const float score = score_buf[i];
      if (score < threshold) [[likely]] {
        continue;
      }
      topk.push(ids[base + i], score);
    }
    threshold = topk.reject_threshold();
  }
}

void prepare_query_into(const ingest::VectorStore& store,
                        const std::optional<transform::SrhtRotation>& rotation,
                        bool query_is_rotated, std::size_t input_dim, std::span<const float> query,
                        PreparedQuery& prepared) {
  const std::size_t srht_dim = store.srht_dim();

  if (prepared.rotated.size() != srht_dim) {
    prepared.rotated.assign(srht_dim, 0.0f);
  }

  if (!store.has_buckets()) {
    throw Error("QueryEngine::prepare requires VectorStore::finalize_buckets");
  }
  const index::BucketIndex& buckets = store.buckets();

  if (query_is_rotated) {
    if (query.size() != srht_dim) {
      throw Error("rotated query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), srht_dim * sizeof(float));
  } else if (rotation.has_value()) {
    if (query.size() != input_dim) {
      throw Error("query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), input_dim * sizeof(float));
    transform::l2_normalize_in_place(std::span<float>(prepared.rotated.data(), input_dim));
    rotation->apply_in_place(std::span<float>(prepared.rotated.data(), srht_dim));
  } else {
    throw Error("QueryEngine requires with_rotation() or from_rotated()");
  }
  // Fold after SRHT (from_rotated queries are already in rotated space).
  prepared.query_bin =
      index::fold_to_bin(buckets.hash(), std::span<const float>(prepared.rotated.data(), srht_dim),
                         buckets.bin_width());
}

}  // namespace

QueryEngine::QueryEngine(const ingest::VectorStore& store,
                         std::optional<transform::SrhtRotation> rotation, bool query_is_rotated,
                         std::size_t input_dim, quantize::LloydMaxCodebook codebook)
    : store_(store),
      rotation_(std::move(rotation)),
      query_is_rotated_(query_is_rotated),
      input_dim_(input_dim),
      codebook_(std::move(codebook)) {}

QueryEngine QueryEngine::with_rotation(const ingest::VectorStore& store, std::size_t input_dim,
                                       std::uint64_t seed) {
  quantize::LloydMaxCodebook codebook(store.srht_dim(), store.bits_per_dim(), store.block_dims());
  return QueryEngine(store, transform::SrhtRotation(input_dim, seed), false, input_dim,
                     std::move(codebook));
}

QueryEngine QueryEngine::from_rotated(const ingest::VectorStore& store) {
  quantize::LloydMaxCodebook codebook(store.srht_dim(), store.bits_per_dim(), store.block_dims());
  return QueryEngine(store, std::nullopt, true, store.input_dim(), std::move(codebook));
}

const BlockedCodes& QueryEngine::blocked_codes() const {
  if (blocked_n_ != store_.size()) {
    blocked_.rebuild(store_.l0_codes(), store_.l0_words_per_vec(), store_.size(), store_.srht_dim(),
                     store_.bits_per_dim(), store_.block_dims());
    blocked_n_ = store_.size();
  }
  return blocked_;
}

void QueryEngine::prepare_index() const { (void)blocked_codes(); }

void QueryEngine::prepare_into(PreparedQuery& out, std::span<const float> query) const {
  prepare_query_into(store_, rotation_, query_is_rotated_, input_dim_, query, out);
}

PreparedQuery QueryEngine::prepare(std::span<const float> query) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return prepared;
}

std::vector<QueryHit> QueryEngine::search_prepared(const PreparedQuery& prepared,
                                                   const QueryParams& params,
                                                   SearchStats* stats) const {
  if (!store_.has_buckets()) {
    throw Error("QueryEngine::search requires VectorStore::finalize_buckets");
  }

  TopKHits topk(params.k);
  const index::BucketIndex& buckets = store_.buckets();
  index::validate_probe_radius(params.probe_radius);

  const BlockedCodes& blocked = blocked_codes();
  const std::size_t m = store_.srht_dim() / store_.block_dims();
  const bool byte_aligned =
      lut_bits_supported(store_.bits_per_dim()) && (m * store_.bits_per_dim()) % 8 == 0;
  build_query_lut(prepared.rotated, codebook_, lut_cache_);
  const bool use_fastscan = !blocked.empty() && byte_aligned && !lut_cache_.empty();

  std::size_t candidates = 0;
  const std::vector<index::BucketRange> ranges =
      buckets.probe(prepared.query_bin, params.probe_radius, &candidates);
  if (stats != nullptr) {
    stats->candidates = candidates;
    stats->cells_probed = ranges.size();
  }

  if (use_fastscan) {
#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
    const bool prefer_parallel =
        ranges.size() >= 8 && candidates >= kParallelMinBlocks * BlockedCodes::kBlock &&
        omp_get_max_threads() > 1;
    if (prefer_parallel) {
      const int nthreads = omp_get_max_threads();
      std::vector<TopKHits>& locals = omp_topk_scratch(params.k, nthreads);
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        TopKHits& local = locals[static_cast<std::size_t>(tid)];
#pragma omp for schedule(dynamic) nowait
        for (int r = 0; r < static_cast<int>(ranges.size()); ++r) {
          const auto& range = ranges[static_cast<std::size_t>(r)];
          score_store_range_fastscan(blocked, lut_cache_, store_, range.start,
                                     range.start + range.length, local);
        }
      }
      for (auto& local : locals) {
        topk.merge_from(local);
      }
      return topk.finalize();
    }
#endif
    for (const auto& range : ranges) {
      score_store_range_fastscan(blocked, lut_cache_, store_, range.start,
                                 range.start + range.length, topk);
    }
    return topk.finalize();
  }

  for (const auto& range : ranges) {
    score_store_range_vector_major(store_, lut_cache_, prepared, codebook_, range.start,
                                   range.start + range.length, topk);
  }
  return topk.finalize();
}

std::vector<QueryHit> QueryEngine::search(std::span<const float> query, const QueryParams& params,
                                          SearchStats* stats) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return search_prepared(prepared, params, stats);
}

}  // namespace vectorcache::query
