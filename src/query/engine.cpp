#include "vectorcache/query/engine.hpp"

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/ingest/store.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

namespace vectorcache::query {

namespace {

constexpr std::size_t kHeapTopKThreshold = 32;
constexpr std::size_t kScoreChunk = 64;

class TopKHits {
 public:
  explicit TopKHits(std::size_t k, std::size_t num_bits)
      : k_(k),
        num_bits_(num_bits),
        use_heap_(k > kHeapTopKThreshold),
        size_(0),
        max_disagree_(0),
        max_idx_(0),
        disagrees_(use_heap_ ? 0 : k, 0),
        ids_(use_heap_ ? 0 : k, 0) {
    if (use_heap_) {
      heap_.reserve(k);
    }
  }

  std::uint32_t reject_threshold() const {
    if (k_ == 0) {
      return 0;
    }
    if (use_heap_) {
      if (heap_.size() < k_) {
        return std::numeric_limits<std::uint32_t>::max();
      }
      return heap_.front().disagree;
    }
    if (size_ < k_) {
      return std::numeric_limits<std::uint32_t>::max();
    }
    return max_disagree_;
  }

  void push(std::size_t id, std::uint32_t disagree) {
    if (k_ == 0) {
      return;
    }
    if (use_heap_) {
      push_heap(id, disagree);
      return;
    }
    if (size_ < k_) {
      disagrees_[size_] = disagree;
      ids_[size_] = id;
      ++size_;
      if (size_ == k_) {
        rescan_max();
      }
      return;
    }
    if (disagree < max_disagree_ || (disagree == max_disagree_ && id < ids_[max_idx_])) {
      disagrees_[max_idx_] = disagree;
      ids_[max_idx_] = id;
      rescan_max();
    }
  }

  std::vector<QueryHit> finalize() const {
    std::vector<QueryHit> out;
    if (use_heap_) {
      out.reserve(heap_.size());
      for (const auto& hit : heap_) {
        out.push_back({hit.id, score_from_disagree(hit.disagree, num_bits_)});
      }
    } else {
      out.reserve(size_);
      for (std::size_t i = 0; i < size_; ++i) {
        out.push_back({ids_[i], score_from_disagree(disagrees_[i], num_bits_)});
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
    std::uint32_t disagree;
    std::size_t id;
  };

  static bool heap_worse(const HeapHit& a, const HeapHit& b) {
    if (a.disagree != b.disagree) {
      return a.disagree < b.disagree;
    }
    return a.id < b.id;
  }

  void push_heap(std::size_t id, std::uint32_t disagree) {
    if (heap_.size() < k_) {
      heap_.push_back({disagree, id});
      std::push_heap(heap_.begin(), heap_.end(), heap_worse);
      return;
    }
    const auto& worst = heap_.front();
    if (disagree < worst.disagree || (disagree == worst.disagree && id < worst.id)) {
      std::pop_heap(heap_.begin(), heap_.end(), heap_worse);
      heap_.back() = {disagree, id};
      std::push_heap(heap_.begin(), heap_.end(), heap_worse);
    }
  }

  void rescan_max() {
    std::size_t mi = 0;
    for (std::size_t h = 1; h < size_; ++h) {
      if (disagrees_[h] > disagrees_[mi] ||
          (disagrees_[h] == disagrees_[mi] && ids_[h] > ids_[mi])) {
        mi = h;
      }
    }
    max_disagree_ = disagrees_[mi];
    max_idx_ = mi;
  }

  std::size_t k_;
  std::size_t num_bits_;
  bool use_heap_;
  std::size_t size_;
  std::uint32_t max_disagree_;
  std::size_t max_idx_;
  std::vector<std::uint32_t> disagrees_;
  std::vector<std::size_t> ids_;
  std::vector<HeapHit> heap_;
};

void search_group(const ingest::ParentGroup& group, const PreparedQuery& query,
                  std::size_t l0_bits, TopKHits& topk) {
  const std::size_t n = group.size();
  if (n == 0) {
    return;
  }

  const std::size_t words = query.l0.size();
  const auto codes = group.l0_codes();
  const auto ids = group.ids();

  alignas(64) std::uint32_t disagree_buf[kScoreChunk];
  std::uint32_t threshold = topk.reject_threshold();
  for (std::size_t base = 0; base < n; base += kScoreChunk) {
    const std::size_t chunk = std::min(kScoreChunk, n - base);
    if (base + chunk + 8 <= n) {
      _mm_prefetch(reinterpret_cast<const char*>(codes.data() + (base + chunk + 4) * words),
                   _MM_HINT_T0);
    }
    bit_agreement_batch_disagree(query.l0, l0_bits, codes.subspan(base * words, chunk * words),
                                 words, chunk, std::span<std::uint32_t>(disagree_buf, chunk),
                                 threshold);
    for (std::size_t i = 0; i < chunk; ++i) {
      if (disagree_buf[i] > threshold) {
        continue;
      }
      topk.push(ids[base + i], disagree_buf[i]);
    }
    // Refresh once per chunk; next chunk sees the tightened reject bound.
    threshold = topk.reject_threshold();
  }
}

void search_with_probe(const ingest::ParentStore& store, const PreparedQuery& query,
                       const QueryParams& params, TopKHits& topk) {
  const std::size_t l0_bits = quantize::l0_bits_per_vector(store.srht_dim());

  auto scan_group = [&](const ingest::ParentGroup* group) {
    if (group == nullptr || group->empty()) {
      return true;
    }
    _mm_prefetch(reinterpret_cast<const char*>(group->l0_codes().data()), _MM_HINT_T0);
    search_group(*group, query, l0_bits, topk);
    return true;
  };

  scan_group(store.find(query.support_key));
  if (params.max_hd < 2) {
    return;
  }

  const auto keys = store.unique_keys();
  const auto& postings = store.dim_postings();

  postings.for_each_hd2(keys, query.support_key,
                        [&](const quantize::SupportKey&, std::uint32_t key_idx) {
                          return scan_group(&store.group_at(key_idx));
                        });

  if (params.max_hd < 4) {
    return;
  }

  thread_local std::vector<std::uint8_t> hd4_visited;
  if (hd4_visited.size() < keys.size()) {
    hd4_visited.resize(keys.size());
  }

  postings.for_each_hd4(keys, query.support_key, hd4_visited,
                        [&](const quantize::SupportKey&, std::uint32_t key_idx) {
                          return scan_group(&store.group_at(key_idx));
                        });
}

void prepare_query_into(const ingest::ParentStore& store,
                        const std::optional<transform::SrhtRotation>& rotation,
                        bool query_is_rotated, std::size_t input_dim,
                        std::span<const float> query, PreparedQuery& prepared) {
  const std::size_t srht_dim = store.srht_dim();
  const std::size_t l0_words = store.l0_words_per_vec();
  const std::size_t top_d = store.top_d();

  if (prepared.rotated.size() != srht_dim) {
    prepared.rotated.assign(srht_dim, 0.0f);
  }
  if (prepared.l0.size() != l0_words) {
    prepared.l0.assign(l0_words, 0);
  }

  if (query_is_rotated) {
    if (query.size() != srht_dim) {
      throw Error("rotated query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), srht_dim * sizeof(float));
    prepared.support_key = quantize::quantize_support_key(
        std::span<const float>(prepared.rotated.data(), store.input_dim()), top_d);
  } else if (rotation.has_value()) {
    if (query.size() != input_dim) {
      throw Error("query dimension mismatch");
    }
    std::memcpy(prepared.rotated.data(), query.data(), input_dim * sizeof(float));
    transform::l2_normalize_in_place(std::span<float>(prepared.rotated.data(), input_dim));
    prepared.support_key = quantize::quantize_support_key(
        std::span<const float>(prepared.rotated.data(), input_dim), top_d);
    if (srht_dim > input_dim) {
      std::memset(prepared.rotated.data() + input_dim, 0, (srht_dim - input_dim) * sizeof(float));
    }
    rotation->apply_in_place(prepared.rotated);
  } else {
    throw Error("QueryEngine requires with_rotation() or from_rotated()");
  }

  quantize::quantize_1dim_to_1bit_into(prepared.rotated, prepared.l0);
}

}  // namespace

QueryEngine::QueryEngine(const ingest::ParentStore& store,
                         std::optional<transform::SrhtRotation> rotation, bool query_is_rotated,
                         std::size_t input_dim)
    : store_(store),
      rotation_(std::move(rotation)),
      query_is_rotated_(query_is_rotated),
      input_dim_(input_dim) {}

QueryEngine QueryEngine::with_rotation(const ingest::ParentStore& store, std::size_t input_dim,
                                       std::uint64_t seed) {
  return QueryEngine(store, transform::SrhtRotation(input_dim, seed), false, input_dim);
}

QueryEngine QueryEngine::from_rotated(const ingest::ParentStore& store) {
  return QueryEngine(store, std::nullopt, true, store.input_dim());
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
  const std::size_t l0_bits = quantize::l0_bits_per_vector(store_.srht_dim());
  TopKHits topk(params.k, l0_bits);
  search_with_probe(store_, prepared, params, topk);
  return topk.finalize();
}

std::vector<QueryHit> QueryEngine::search(std::span<const float> query,
                                          const QueryParams& params) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return search_prepared(prepared, params);
}

}  // namespace vectorcache::query
