#include "vectorcache/query/engine.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/query/distance.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

namespace vectorcache::query {

namespace {

constexpr std::size_t kHeapTopKThreshold = 32;
constexpr std::size_t kScoreChunk = 64;

float parent_bit_agreement(std::uint8_t a, std::uint8_t b) {
  const unsigned disagree = static_cast<unsigned>(std::popcount(static_cast<unsigned>(a ^ b)));
  const unsigned agree = static_cast<unsigned>(quantize::PARENT_BITS) - disagree;
  return (2.0f * static_cast<float>(agree) - static_cast<float>(quantize::PARENT_BITS)) /
         static_cast<float>(quantize::PARENT_BITS);
}

/// Top-k by integer disagreement (lower is better). Float scores only at finalize.
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

  /// Worst disagree currently in the set; UINT32_MAX until full (accept everything).
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
    // Better = lower disagree; on tie prefer smaller id (replace if id < worst id).
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

  /// Max-heap by disagree (worst at front); on equal disagree, larger id is worse.
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

std::uint8_t find_best_parent(std::uint8_t query_key, std::span<const std::uint8_t> keys) {
  if (keys.empty()) {
    throw Error("ParentStore has no unique parent keys");
  }
  std::uint8_t best = keys[0];
  float best_score = parent_bit_agreement(query_key, best);
  for (std::size_t i = 1; i < keys.size(); ++i) {
    const float score = parent_bit_agreement(query_key, keys[i]);
    if (score > best_score) {
      best_score = score;
      best = keys[i];
    }
  }
  return best;
}

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
  for (std::size_t base = 0; base < n; base += kScoreChunk) {
    const std::size_t chunk = std::min(kScoreChunk, n - base);
    if (base + chunk + 8 <= n) {
      _mm_prefetch(reinterpret_cast<const char*>(codes.data() + (base + chunk + 4) * words),
                   _MM_HINT_T0);
    }
    bit_agreement_batch_disagree(query.l0, l0_bits, codes.subspan(base * words, chunk * words),
                                 words, chunk, std::span<std::uint32_t>(disagree_buf, chunk));
    for (std::size_t i = 0; i < chunk; ++i) {
      if (disagree_buf[i] > topk.reject_threshold()) {
        continue;
      }
      topk.push(ids[base + i], disagree_buf[i]);
    }
  }
}

void search_store(const ingest::ParentStore& store, const PreparedQuery& query, TopKHits& topk) {
  const std::uint8_t best = find_best_parent(query.parent_key, store.unique_keys());
  const std::size_t l0_bits = quantize::l0_bits_per_vector(store.padded_dim());

  std::array<std::uint8_t, quantize::PARENT_BITS + 1> probe_keys{};
  probe_keys[0] = best;
  for (std::size_t i = 0; i < quantize::PARENT_BITS; ++i) {
    probe_keys[i + 1] = static_cast<std::uint8_t>(best ^ static_cast<std::uint8_t>(1u << i));
  }

  for (const std::uint8_t key : probe_keys) {
    const ingest::ParentGroup* group = store.group_for_key(key);
    if (group != nullptr && !group->empty()) {
      search_group(*group, query, l0_bits, topk);
    }
  }
}

void prepare_query_into(const ingest::ParentStore& store,
                        const std::optional<transform::SrhtRotation>& rotation,
                        bool query_is_rotated, std::size_t input_dim,
                        std::span<const float> query, PreparedQuery& prepared) {
  const std::size_t padded_dim = store.padded_dim();
  const std::size_t l0_words = store.l0_words_per_vec();

  if (prepared.rotated.size() != padded_dim) {
    prepared.rotated.assign(padded_dim, 0.0f);
  }
  if (prepared.l0.size() != l0_words) {
    prepared.l0.assign(l0_words, 0);
  }

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

  prepared.parent_key = quantize::quantize_parent_8bit(prepared.rotated);
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
  return QueryEngine(store, std::nullopt, true, store.padded_dim());
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
  const std::size_t l0_bits = quantize::l0_bits_per_vector(store_.padded_dim());
  TopKHits topk(params.k, l0_bits);
  search_store(store_, prepared, topk);
  return topk.finalize();
}

std::vector<QueryHit> QueryEngine::search(std::span<const float> query,
                                          const QueryParams& params) const {
  PreparedQuery prepared;
  prepare_into(prepared, query);
  return search_prepared(prepared, params);
}

}  // namespace vectorcache::query
