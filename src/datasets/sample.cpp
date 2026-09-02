#include "vectorcache/datasets/sample.hpp"

#include <algorithm>
#include <stdexcept>

#include "vectorcache/error.hpp"

namespace vectorcache::datasets {

std::vector<std::size_t> sample_index_range(std::size_t start, std::size_t end, std::size_t count,
                                            std::uint64_t seed) {
  if (start >= end) {
    throw Error("sample_index_range: empty range");
  }
  const std::size_t available = end - start;
  if (count > available) {
    throw Error("sample_index_range: count exceeds range");
  }

  std::vector<std::size_t> pool(available);
  for (std::size_t i = 0; i < available; ++i) {
    pool[i] = start + i;
  }

  std::mt19937_64 rng(seed);
  for (std::size_t i = 0; i < count; ++i) {
    const std::size_t j = i + static_cast<std::size_t>(rng() % (available - i));
    std::swap(pool[i], pool[j]);
  }
  pool.resize(count);
  std::sort(pool.begin(), pool.end());
  return pool;
}

IndexSubsetReader::IndexSubsetReader(DatasetReader& inner, std::vector<std::size_t> indices)
    : inner_(inner), indices_(std::move(indices)) {
  std::sort(indices_.begin(), indices_.end());
  if (!indices_.empty()) {
    next_want_ = indices_.front();
  }
}

DatasetMeta IndexSubsetReader::meta() const {
  auto m = inner_.meta();
  m.count = indices_.size();
  return m;
}

bool IndexSubsetReader::next_vector_into(std::span<float> out) {
  if (pos_ >= indices_.size()) {
    return false;
  }

  const std::size_t target = indices_[pos_];
  while (stream_idx_ < target) {
    if (!inner_.next_vector_into(out)) {
      throw Error("IndexSubsetReader: stream ended before target index");
    }
    ++stream_idx_;
  }

  if (!inner_.next_vector_into(out)) {
    throw Error("IndexSubsetReader: failed to read target index");
  }
  ++stream_idx_;
  ++pos_;
  if (pos_ < indices_.size()) {
    next_want_ = indices_[pos_];
  }
  return true;
}

LimitedReader::LimitedReader(DatasetReader& inner, std::size_t limit)
    : inner_(inner), remaining_(limit) {}

DatasetMeta LimitedReader::meta() const {
  auto m = inner_.meta();
  m.count = std::min(m.count, remaining_);
  return m;
}

bool LimitedReader::next_vector_into(std::span<float> out) {
  if (remaining_ == 0) {
    return false;
  }
  if (!inner_.next_vector_into(out)) {
    throw Error("LimitedReader: stream ended early");
  }
  --remaining_;
  return true;
}

}  // namespace vectorcache::datasets
