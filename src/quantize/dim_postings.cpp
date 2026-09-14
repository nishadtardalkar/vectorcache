#include "vectorcache/quantize/dim_postings.hpp"

#include <algorithm>

#include "vectorcache/error.hpp"

namespace vectorcache::quantize {

DimPostingIndex DimPostingIndex::build(std::span<const SupportKey> keys, std::size_t input_dim) {
  DimPostingIndex idx;
  idx.num_keys_ = keys.size();
  idx.postings_.assign(input_dim, {});
  for (std::size_t ki = 0; ki < keys.size(); ++ki) {
    const SupportKey& key = keys[ki];
    if (key.d == 0) {
      continue;
    }
    const auto key_idx = static_cast<std::uint32_t>(ki);
    for (std::uint8_t i = 0; i < key.d; ++i) {
      const std::uint16_t dim = key.dims[i];
      if (dim >= input_dim) {
        throw Error("DimPostingIndex: dim out of range");
      }
      idx.postings_[dim].push_back(key_idx);
    }
  }
  for (auto& list : idx.postings_) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }
  return idx;
}

}  // namespace vectorcache::quantize
