#include "vectorcache/ingest/engine.hpp"

#include <algorithm>
#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"
#include "vectorcache/transform/normalize.hpp"

namespace vectorcache::ingest {

IngestionEngine::IngestionEngine(BlockStore store, std::optional<transform::SrhtRotation> rotation,
                                 bool quantize_only, std::size_t input_dim, std::size_t padded_dim,
                                 std::size_t l1_words_per_vec, std::size_t l0_words_per_vec)
    : store_(std::move(store)),
      rotation_(std::move(rotation)),
      quantize_only_(quantize_only),
      input_dim_(input_dim),
      padded_dim_(padded_dim),
      l1_words_per_vec_(l1_words_per_vec),
      l0_words_per_vec_(l0_words_per_vec),
      read_buf_(input_dim, 0.0f) {}

IngestionEngine IngestionEngine::from_rotated(std::size_t padded_dim) {
  const std::size_t l1_words = quantize::l1_words_per_vector(padded_dim);
  const std::size_t l0_words = quantize::l0_words_per_vector(padded_dim);
  return IngestionEngine(BlockStore(l1_words, l0_words, padded_dim), std::nullopt, true,
                         padded_dim, padded_dim, l1_words, l0_words);
}

IngestionEngine IngestionEngine::with_rotation(std::size_t original_dim, std::uint64_t seed) {
  transform::SrhtRotation rotation(original_dim, seed);
  const std::size_t padded = rotation.padded_dim();
  const std::size_t l1_words = quantize::l1_words_per_vector(padded);
  const std::size_t l0_words = quantize::l0_words_per_vector(padded);
  return IngestionEngine(BlockStore(l1_words, l0_words, padded), std::move(rotation), false,
                         original_dim, padded, l1_words, l0_words);
}

void IngestionEngine::reserve_vectors(std::size_t count) {
  store_ = BlockStore::with_capacity(l1_words_per_vec_, l0_words_per_vec_, padded_dim_, count);
  ensure_batch_capacity(std::min(INGEST_BATCH_SIZE, std::max(count, std::size_t{1})));
}

const transform::SrhtRotation* IngestionEngine::rotation() const {
  return rotation_ ? &(*rotation_) : nullptr;
}

void IngestionEngine::ensure_batch_capacity(std::size_t batch_cap) {
  if (batch_work_.size() < batch_cap) {
    batch_work_.resize(batch_cap);
    for (auto& work : batch_work_) {
      work.rotated.assign(padded_dim_, 0.0f);
      work.l1.assign(l1_words_per_vec_, 0);
      work.l0.assign(l0_words_per_vec_, 0);
    }
  }
  const std::size_t input_bytes = batch_cap * input_dim_;
  if (batch_inputs_.size() < input_bytes) {
    batch_inputs_.resize(input_bytes, 0.0f);
  }
}

std::size_t IngestionEngine::read_batch(datasets::DatasetReader& reader) {
  const std::size_t max_batch = batch_work_.size();
  std::size_t count = 0;
  while (count < max_batch) {
    if (!reader.next_vector_into(read_buf_)) {
      break;
    }
    const std::size_t dst_start = count * input_dim_;
    std::copy(read_buf_.begin(), read_buf_.end(),
              batch_inputs_.begin() + static_cast<std::ptrdiff_t>(dst_start));
    ++count;
  }
  return count;
}

void IngestionEngine::process_batch(std::size_t batch_len) {
  const std::size_t input_dim = input_dim_;
  const std::size_t padded_dim = padded_dim_;
  const bool quantize_only = quantize_only_;
  const bool has_rotation = rotation_.has_value();
  const transform::SrhtRotation* rotation = has_rotation ? &(*rotation_) : nullptr;

  for (std::size_t i = 0; i < batch_len; ++i) {
    auto& work = batch_work_[i];
    const float* input = batch_inputs_.data() + static_cast<std::ptrdiff_t>(i * input_dim);
    if (has_rotation) {
      std::memcpy(work.rotated.data(), input, input_dim * sizeof(float));
      if (padded_dim > input_dim) {
        std::memset(work.rotated.data() + input_dim, 0,
                    (padded_dim - input_dim) * sizeof(float));
      }
      transform::l2_normalize_in_place(work.rotated);
      rotation->apply_in_place(work.rotated);
    } else if (quantize_only) {
      std::memcpy(work.rotated.data(), input, input_dim * sizeof(float));
    } else {
      throw Error("IngestionEngine requires with_rotation() or from_rotated()");
    }
    quantize::quantize_4d_to_1bit_into(work.rotated, work.l1);
    quantize::quantize_1dim_to_1bit_into(work.rotated, work.l0);
  }
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
        hook->on_vector(global_id, work.rotated);
      }
      store_.push_vector(work.l1, work.l0);
      ++global_id;
    }
  }

  return IngestReport{store_.block_count(), store_.partial_block().len(), global_id};
}

}  // namespace vectorcache::ingest
