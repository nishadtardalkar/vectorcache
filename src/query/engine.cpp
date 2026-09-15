#include "vectorcache/query/engine.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

namespace vectorcache::query {

namespace {

constexpr std::size_t kHeapTopKThreshold = 32;
constexpr std::size_t kScoreChunk = 512;

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

  std::vector<QueryHit> finalize() const {
    std::vector<QueryHit> out;
    if (use_heap_) {
      out.reserve(heap_.size());
      for (const auto& hit : heap_) {
        out.push_back({hit.id, hit.score});
      }
    } else {
      out.reserve(size_);
      for (std::size_t i = 0; i < size_; ++i) {
        out.push_back({ids_[i], scores_[i]});
      }
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

  /// Max-heap of the worst (lowest) scores among the current top-k.
  static bool heap_worse(const HeapHit& a, const HeapHit& b) {
    if (a.score != b.score) {
      return a.score > b.score;  // better (higher) score is "less"
    }
    return a.id < b.id;  // smaller id is better when scores tie
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

void search_flat(const ingest::VectorStore& store, const PreparedQuery& query,
                 const quantize::LloydMaxCodebook& codebook, TopKHits& topk) {
  const std::size_t n = store.size();
  if (n == 0) {
    return;
  }

  const std::size_t words = store.l0_words_per_vec();
  const auto codes = store.l0_codes();
  const auto ids = store.ids();

  QueryLut lut;
  build_query_lut(query.rotated, codebook, lut);

  alignas(64) float score_buf[kScoreChunk];
  float threshold = topk.reject_threshold();
  for (std::size_t base = 0; base < n; base += kScoreChunk) {
    const std::size_t chunk = std::min(kScoreChunk, n - base);
    asymmetric_ip_batch_lut(lut, codes.subspan(base * words, chunk * words), words, chunk, codebook,
                            query.rotated, std::span<float>(score_buf, chunk));
    for (std::size_t i = 0; i < chunk; ++i) {
      if (score_buf[i] < threshold) {
        continue;
      }
      topk.push(ids[base + i], score_buf[i]);
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
    if (srht_dim > input_dim) {
      std::memset(prepared.rotated.data() + input_dim, 0, (srht_dim - input_dim) * sizeof(float));
    }
    rotation->apply_in_place(prepared.rotated);
  } else {
    throw Error("QueryEngine requires with_rotation() or from_rotated()");
  }
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
  quantize::LloydMaxCodebook codebook(store.srht_dim(), store.bits_per_dim());
  return QueryEngine(store, transform::SrhtRotation(input_dim, seed), false, input_dim,
                     std::move(codebook));
}

QueryEngine QueryEngine::from_rotated(const ingest::VectorStore& store) {
  quantize::LloydMaxCodebook codebook(store.srht_dim(), store.bits_per_dim());
  return QueryEngine(store, std::nullopt, true, store.input_dim(), std::move(codebook));
}

void QueryEngine::prepare_into(PreparedQuery& out, std::span<const float> query) const {
  prepare_query_into(store_, rotation_, query_is_rotated_, input_dim_, query, out);
}

PreparedQuery QueryEngine::prepare(std::span<const float> query) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return prepared;
}

std::vector<QueryHit> QueryEngine::search_prepared(const PreparedQuery& prepared,
                                                   const QueryParams& params) const {
  TopKHits topk(params.k);
  search_flat(store_, prepared, codebook_, topk);
  return topk.finalize();
}

std::vector<QueryHit> QueryEngine::search(std::span<const float> query,
                                          const QueryParams& params) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return search_prepared(prepared, params);
}

}  // namespace vectorcache::query
