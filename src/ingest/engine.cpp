#include "vectorcache/ingest/engine.hpp"

#include <algorithm>
#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
#include <omp.h>
#endif

namespace vectorcache::ingest {

IngestionEngine::IngestionEngine(VectorStore store, std::optional<transform::SrhtRotation> rotation,
                                 bool quantize_only, std::size_t input_dim, std::size_t srht_dim,
                                 std::size_t l0_words_per_vec)
    : store_(std::move(store)),
      rotation_(std::move(rotation)),
      quantize_only_(quantize_only),
      input_dim_(input_dim),
      srht_dim_(srht_dim),
      l0_words_per_vec_(l0_words_per_vec) {}

IngestionEngine IngestionEngine::from_rotated(std::size_t srht_dim) {
  const std::size_t l0_words = quantize::l0_words_per_vector(srht_dim);
  return IngestionEngine(VectorStore(l0_words, srht_dim, srht_dim), std::nullopt, true, srht_dim,
                         srht_dim, l0_words);
}

IngestionEngine IngestionEngine::with_rotation(std::size_t original_dim, std::uint64_t seed) {
  transform::SrhtRotation rotation(original_dim, seed);
  const std::size_t srht = rotation.srht_dim();
  const std::size_t l0_words = quantize::l0_words_per_vector(srht);
  return IngestionEngine(VectorStore(l0_words, original_dim, srht), std::move(rotation), false,
                         original_dim, srht, l0_words);
}

void IngestionEngine::reserve_vectors(std::size_t count) {
  store_ = VectorStore::with_capacity(l0_words_per_vec_, input_dim_, srht_dim_, count);
  ensure_batch_capacity(std::min(INGEST_BATCH_SIZE, std::max(count, std::size_t{1})));
}

void IngestionEngine::ensure_batch_capacity(std::size_t batch_cap) {
  if (batch_work_.size() < batch_cap) {
    batch_work_.resize(batch_cap);
    for (auto& work : batch_work_) {
      work.buf.assign(srht_dim_, 0.0f);
      work.l0.assign(l0_words_per_vec_, 0);
    }
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
  const bool quantize_only = quantize_only_;
  const bool has_rotation = rotation_.has_value();
  if (!has_rotation && !quantize_only) {
    throw Error("IngestionEngine requires with_rotation() or from_rotated()");
  }
  const transform::SrhtRotation* rotation = has_rotation ? &(*rotation_) : nullptr;
  const std::size_t input_dim = input_dim_;
  const std::size_t srht_dim = srht_dim_;

#if defined(VECTORCACHE_OPENMP) && VECTORCACHE_OPENMP
#pragma omp parallel for schedule(static)
  for (int i = 0; i < static_cast<int>(batch_len); ++i) {
    auto& work = batch_work_[static_cast<std::size_t>(i)];
    if (has_rotation) {
      transform::l2_normalize_in_place(std::span<float>(work.buf.data(), input_dim));
      if (srht_dim > input_dim) {
        std::memset(work.buf.data() + input_dim, 0, (srht_dim - input_dim) * sizeof(float));
      }
      rotation->apply_in_place(work.buf);
    }
    quantize::quantize_1dim_to_1bit_into(work.buf, work.l0);
  }
#else
  for (std::size_t i = 0; i < batch_len; ++i) {
    auto& work = batch_work_[i];
    if (has_rotation) {
      transform::l2_normalize_in_place(std::span<float>(work.buf.data(), input_dim));
      if (srht_dim > input_dim) {
        std::memset(work.buf.data() + input_dim, 0, (srht_dim - input_dim) * sizeof(float));
      }
      rotation->apply_in_place(work.buf);
    }
    quantize::quantize_1dim_to_1bit_into(work.buf, work.l0);
  }
#endif
}

IngestReport IngestionEngine::ingest(datasets::DatasetReader& reader) {
  return ingest_with_hook(reader, nullptr);
}

IngestReport IngestionEngine::ingest_with_hook(datasets::DatasetReader& reader, VectorHook* hook) {
  const std::size_t meta_count = reader.meta().count;
  if (meta_count > 0) {
    reserve_vectors(meta_count);
  } else {
    ensure_batch_capacity(INGEST_BATCH_SIZE);
  }

  std::uint64_t global_id = 0;

  while (true) {
    const std::size_t batch_len = read_batch(reader);
    if (batch_len == 0) {
      break;
    }

    process_batch(batch_len);

    for (std::size_t i = 0; i < batch_len; ++i) {
      const auto& work = batch_work_[i];
      if (hook != nullptr) {
        hook->on_vector(global_id, work.buf);
      }
      store_.push(static_cast<std::size_t>(global_id), work.l0);
      ++global_id;
    }
  }

  return IngestReport{global_id};
}

}  // namespace vectorcache::ingest
