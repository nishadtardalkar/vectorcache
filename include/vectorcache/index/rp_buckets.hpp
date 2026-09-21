#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"

namespace vectorcache::index {

inline constexpr std::size_t kMaxPairDirs = 64;
inline constexpr std::size_t kMaxProbeCells = 4096;

/// Default ridge δ = 1/dim for post-SRHT fold.
inline float default_fold_ridge(std::size_t dim) {
  return 1.0f / static_cast<float>(dim > 0 ? dim : 1);
}

/// Cyclic list of L random unit vectors in R^2 for recursive pairwise folding.
class PairHash {
 public:
  PairHash() = default;
  /// `ridge_delta` must be finite and > 0 (used as δ in ⟨(a,b),r⟩ / √(a²+b²+δ)).
  PairHash(std::size_t num_pair_dirs, std::uint64_t seed, float ridge_delta);

  std::size_t num_pair_dirs() const { return num_pair_dirs_; }
  float ridge_delta() const { return ridge_delta_; }
  bool empty() const { return num_pair_dirs_ == 0; }

  /// Recursive pairwise reduce to one scalar in [-1, 1]. Cursor starts at 0 each call.
  float fold(std::span<const float> x) const;

 private:
  std::size_t num_pair_dirs_ = 0;
  float ridge_delta_ = 0.0f;
  /// Packed (r0, r1) pairs, length 2 * num_pair_dirs_.
  AlignedVector<float> dirs_;
};

/// fold → u = (clamp(s)+1)/2 → floor(u / bin_width). Uniform bins on s ∈ [-1,1].
std::int32_t fold_to_bin(const PairHash& hash, std::span<const float> x, float bin_width);

/// Pack a signed 1D bin into a cell key (two's-complement cast).
std::uint64_t pack_bin(std::int32_t bin);

/// Validate 1D probe count 2P+1 <= kMaxProbeCells.
void validate_probe_radius(std::size_t probe_radius);

struct BucketRange {
  std::size_t start = 0;
  std::size_t length = 0;
};

/// CSR over packed cell keys. Ranges are contiguous store slices (store was permuted by key).
class BucketIndex {
 public:
  BucketIndex() = default;

  /// `sorted_keys[i]` is the cell key of store row i after argsort-by-key permute.
  static BucketIndex build(std::span<const std::uint64_t> sorted_keys, PairHash hash,
                           float bin_width);

  bool empty() const { return keys_.empty(); }
  std::size_t num_cells() const { return keys_.size(); }
  float bin_width() const { return bin_width_; }
  const PairHash& hash() const { return hash_; }

  /// Contiguous CSR range for cell index `i` in key order (0 .. num_cells()-1).
  BucketRange cell(std::size_t i) const;

  /// Find contiguous range for an exact cell key; length 0 if missing.
  BucketRange find(std::uint64_t key) const;

  /// Multi-probe: cells with |offset| <= P, ordered by |offset| then signed offset.
  /// Only non-empty cells are returned. `out_candidates` sums range lengths when non-null.
  std::vector<BucketRange> probe(std::int32_t query_bin, std::size_t probe_radius,
                                 std::size_t* out_candidates = nullptr) const;

 private:
  std::vector<std::uint64_t> keys_;
  std::vector<std::size_t> offsets_;  // size keys_+1
  PairHash hash_;
  float bin_width_ = 0.0f;
};

}  // namespace vectorcache::index
