#include "vectorcache/ingest/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
#include <omp.h>
#endif

namespace vectorcache::ingest {
namespace {

std::uint64_t resolve_bucket_seed(std::uint64_t bucket_seed, std::uint64_t rotation_seed,
                                  bool has_rotation_seed) {
  if (bucket_seed != 0) {
    return bucket_seed;
  }
  if (has_rotation_seed) {
    // SplitMix64-style mix so bucket seed differs from SRHT seed.
    std::uint64_t z = rotation_seed + 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  return 0xC0FFEE5EEDULL;
}

}  // namespace

IngestionEngine::IngestionEngine(VectorStore store, std::optional<transform::SrhtRotation> rotation,
                                 bool quantize_only, std::size_t input_dim, std::size_t srht_dim,
                                 std::size_t l0_words_per_vec, quantize::LloydMaxCodebook codebook,
                                 BucketParams bucket_params, std::uint64_t resolved_bucket_seed)
    : store_(std::move(store)),
      rotation_(std::move(rotation)),
      quantize_only_(quantize_only),
      input_dim_(input_dim),
      srht_dim_(srht_dim),
      l0_words_per_vec_(l0_words_per_vec),
      codebook_(std::move(codebook)),
      bucket_params_(bucket_params),
      bucket_seed_(resolved_bucket_seed),
      centroids_(bucket_params.num_buckets, srht_dim, resolved_bucket_seed) {
  if (bucket_params.num_buckets == 0 || bucket_params.num_buckets > index::kMaxBuckets) {
    throw Error("BucketParams.num_buckets must be in 1..kMaxBuckets");
  }
}

IngestionEngine IngestionEngine::from_rotated(std::size_t srht_dim, std::size_t bits_per_dim,
                                              BucketParams buckets) {
  quantize::LloydMaxCodebook codebook(srht_dim, bits_per_dim);
  const std::size_t l0_words = quantize::l0_words_per_vector(srht_dim, bits_per_dim);
  const std::uint64_t seed = resolve_bucket_seed(buckets.bucket_seed, 0, false);
  return IngestionEngine(VectorStore(l0_words, srht_dim, srht_dim, bits_per_dim), std::nullopt, true,
                         srht_dim, srht_dim, l0_words, std::move(codebook), buckets, seed);
}

IngestionEngine IngestionEngine::with_rotation(std::size_t original_dim, std::uint64_t seed,
                                               std::size_t bits_per_dim, BucketParams buckets) {
  transform::SrhtRotation rotation(original_dim, seed);
  const std::size_t srht = rotation.srht_dim();
  quantize::LloydMaxCodebook codebook(srht, bits_per_dim);
  const std::size_t l0_words = quantize::l0_words_per_vector(srht, bits_per_dim);
  const std::uint64_t bucket_seed = resolve_bucket_seed(buckets.bucket_seed, seed, true);
  return IngestionEngine(VectorStore(l0_words, original_dim, srht, bits_per_dim), std::move(rotation),
                         false, original_dim, srht, l0_words, std::move(codebook), buckets,
                         bucket_seed);
}

void IngestionEngine::reserve_vectors(std::size_t count) {
  store_ = VectorStore::with_capacity(l0_words_per_vec_, input_dim_, srht_dim_, count,
                                      codebook_.bits());
  ensure_batch_capacity(std::min(INGEST_BATCH_SIZE, std::max(count, std::size_t{1})));
  rotated_all_.reserve(count * srht_dim_);
}

void IngestionEngine::ensure_batch_capacity(std::size_t batch_cap) {
  if (batch_work_.size() < batch_cap) {
    batch_work_.resize(batch_cap);
  }
  for (auto& work : batch_work_) {
    if (work.buf.size() != srht_dim_) {
      work.buf.assign(srht_dim_, 0.0f);
    }
    if (work.l0.size() != l0_words_per_vec_) {
      work.l0.assign(l0_words_per_vec_, 0);
    }
    work.alpha = 1.0f;
  }
}

std::size_t IngestionEngine::read_batch(datasets::DatasetReader& reader) {
  const std::size_t max_batch = batch_work_.size();
  const std::size_t input_dim = input_dim_;
  std::size_t count = 0;
  while (count < max_batch) {
    auto& work = batch_work_[count];
    if (!reader.next_vector_into(std::span<float>(work.buf.data(), input_dim))) {
      break;
    }
    ++count;
  }
  return count;
}

void IngestionEngine::process_batch(std::size_t batch_len) {
  const bool has_rotation = rotation_.has_value();
  if (!has_rotation && !quantize_only_) {
    throw Error("IngestionEngine requires with_rotation() or from_rotated()");
  }
  const transform::SrhtRotation* rotation = has_rotation ? &(*rotation_) : nullptr;
  const std::size_t input_dim = input_dim_;
  const std::size_t srht_dim = srht_dim_;
  const quantize::LloydMaxCodebook* codebook = &codebook_;

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(batch_len); ++i) {
    auto& work = batch_work_[static_cast<std::size_t>(i)];
    if (has_rotation) {
      transform::l2_normalize_in_place(std::span<float>(work.buf.data(), input_dim));
      rotation->apply_in_place(std::span<float>(work.buf.data(), srht_dim));
    }
    quantize::quantize_blocks_to_nbit_into(work.buf, *codebook, work.l0);
    work.alpha = quantize::ip_scale_alpha(work.buf, work.l0, *codebook);
  }
#else
  for (std::size_t i = 0; i < batch_len; ++i) {
    auto& work = batch_work_[i];
    if (has_rotation) {
      transform::l2_normalize_in_place(std::span<float>(work.buf.data(), input_dim));
      rotation->apply_in_place(std::span<float>(work.buf.data(), srht_dim));
    }
    quantize::quantize_blocks_to_nbit_into(work.buf, *codebook, work.l0);
    work.alpha = quantize::ip_scale_alpha(work.buf, work.l0, *codebook);
  }
#endif
}

void IngestionEngine::maybe_rebalance(std::vector<std::uint64_t>& cell_keys, bool force) {
  if (cell_keys.empty()) {
    return;
  }
  const std::size_t n = cell_keys.size();
  if (!force) {
    const std::size_t every = bucket_params_.rebalance_every;
    if (every == 0 || (n % every) != 0) {
      return;
    }
  }
  centroids_.rebalance(std::span<const float>(rotated_all_.data(), n * srht_dim_),
                       std::span<std::uint64_t>(cell_keys.data(), n), bucket_params_.ortho_eta);
}

void IngestionEngine::finalize_bucket_index(std::vector<std::uint64_t>& cell_keys) {
  maybe_rebalance(cell_keys, /*force=*/true);
  store_.finalize_buckets(cell_keys, std::move(centroids_));
  rotated_all_.clear();
  rotated_all_.shrink_to_fit();
}

IngestReport IngestionEngine::ingest(datasets::DatasetReader& reader, bool finalize_buckets) {
  return ingest_with_hook(reader, nullptr, finalize_buckets);
}

IngestReport IngestionEngine::ingest_with_hook(datasets::DatasetReader& reader, VectorHook* hook,
                                               bool finalize_buckets) {
  const std::size_t meta_count = reader.meta().count;
  if (meta_count > 0) {
    reserve_vectors(meta_count);
  } else {
    ensure_batch_capacity(INGEST_BATCH_SIZE);
  }

  std::uint64_t global_id = 0;
  std::vector<std::uint64_t> cell_keys;
  if (meta_count > 0) {
    cell_keys.reserve(meta_count);
  }

  while (true) {
    const std::size_t batch_len = read_batch(reader);
    if (batch_len == 0) {
      break;
    }

    process_batch(batch_len);

    for (std::size_t i = 0; i < batch_len; ++i) {
      auto& work = batch_work_[i];
      if (hook != nullptr) {
        hook->on_vector(global_id, work.buf);
      }

      // Online cluster assign + retain post-SRHT float for rebalance.
      const std::uint64_t cell_key = centroids_.assign_and_update(
          std::span<const float>(work.buf.data(), srht_dim_));
      const std::size_t off = rotated_all_.size();
      rotated_all_.resize(off + srht_dim_);
      std::memcpy(rotated_all_.data() + off, work.buf.data(), srht_dim_ * sizeof(float));
      cell_keys.push_back(cell_key);

      store_.push(static_cast<std::size_t>(global_id), work.l0, work.alpha);
      ++global_id;

      maybe_rebalance(cell_keys, /*force=*/false);
    }
  }

  if (global_id == 0) {
    return IngestReport{0};
  }
  if (finalize_buckets) {
    if (cell_keys.size() != static_cast<std::size_t>(global_id)) {
      throw Error("ingest: cell key count mismatch");
    }
    finalize_bucket_index(cell_keys);
  }
  return IngestReport{global_id};
}

}  // namespace vectorcache::ingest
