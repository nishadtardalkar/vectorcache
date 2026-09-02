#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "vectorcache/datasets/reader.hpp"

namespace vectorcache::datasets {

/// Fisher-Yates shuffle of [start, end); returns first count elements.
std::vector<std::size_t> sample_index_range(std::size_t start, std::size_t end, std::size_t count,
                                            std::uint64_t seed);

/// Reads vectors whose stream index is in index_set (must be sorted ascending for efficiency).
class IndexSubsetReader : public DatasetReader {
 public:
  IndexSubsetReader(DatasetReader& inner, std::vector<std::size_t> indices);

  DatasetMeta meta() const override;
  bool next_vector_into(std::span<float> out) override;

 private:
  DatasetReader& inner_;
  std::vector<std::size_t> indices_;
  std::size_t pos_ = 0;
  std::size_t stream_idx_ = 0;
  std::size_t next_want_ = 0;
};

/// Pass-through for first limit vectors, then stop.
class LimitedReader : public DatasetReader {
 public:
  LimitedReader(DatasetReader& inner, std::size_t limit);

  DatasetMeta meta() const override;
  bool next_vector_into(std::span<float> out) override;

 private:
  DatasetReader& inner_;
  std::size_t remaining_;
};

}  // namespace vectorcache::datasets
