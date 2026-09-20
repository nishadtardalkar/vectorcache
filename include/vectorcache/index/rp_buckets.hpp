#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectorcache/aligned.hpp"

namespace vectorcache::index {

inline constexpr std::size_t kMaxProjections = 8;
inline constexpr std::size_t kMaxProbeCells = 4096;

/// R independent unit vectors in R^{dim} (row-major: R * dim floats).
class ProjectionMatrix {
 public:
  ProjectionMatrix() = default;
  ProjectionMatrix(std::size_t num_projections, std::size_t dim, std::uint64_t seed);

  std::size_t num_projections() const { return num_projections_; }
  std::size_t dim() const { return dim_; }
  bool empty() const { return num_projections_ == 0; }

  /// out[r] = ⟨row_r, x⟩. out.size() == num_projections_.
  void project(std::span<const float> x, std::span<float> out) const;

  std::span<const float> row(std::size_t r) const;

 private:
  std::size_t num_projections_ = 0;
  std::size_t dim_ = 0;
  AlignedVector<float> data_;
};

struct BinCodec {
  float bin_width = 0.0f;
  std::int32_t bin_lo = 0;
  std::uint32_t bits_per_axis = 0;
  std::size_t num_projections = 0;

  bool packable() const { return bits_per_axis > 0 && bits_per_axis * num_projections <= 64; }
};

/// bin = floor(proj / w). For unit vectors, proj ∈ [-1, 1].
void project_to_bins(const ProjectionMatrix& matrix, std::span<const float> x, float bin_width,
                     std::span<std::int32_t> bins);

BinCodec make_bin_codec(std::size_t num_projections, float bin_width);

std::uint64_t pack_cell_key(std::span<const std::int32_t> bins, const BinCodec& codec);
/// Same packing as pack_cell_key without range checks (bins must be in codec range).
std::uint64_t pack_cell_key_unchecked(std::span<const std::int32_t> bins, const BinCodec& codec);
void unpack_cell_key(std::uint64_t key, const BinCodec& codec, std::span<std::int32_t> bins);

/// Validate probe grid size (2P+1)^R <= kMaxProbeCells.
void validate_probe_grid(std::size_t num_projections, std::size_t probe_radius);

struct BucketRange {
  std::size_t start = 0;
  std::size_t length = 0;
};

/// Contiguous CSR over store indices sorted by packed cell key.
class BucketIndex {
 public:
  BucketIndex() = default;

  /// `sorted_keys[i]` is the cell key of store row i after argsort-by-key permute.
  static BucketIndex build(std::span<const std::uint64_t> sorted_keys, ProjectionMatrix matrix,
                           BinCodec codec);

  bool empty() const { return keys_.empty(); }
  std::size_t num_cells() const { return keys_.size(); }
  std::size_t num_projections() const { return codec_.num_projections; }
  float bin_width() const { return codec_.bin_width; }
  const ProjectionMatrix& matrix() const { return matrix_; }
  const BinCodec& codec() const { return codec_; }

  /// Contiguous store range for cell index `i` in key order (0 .. num_cells()-1).
  BucketRange cell(std::size_t i) const;

  /// Find contiguous range for an exact cell key; length 0 if missing.
  BucketRange find(std::uint64_t key) const;

  /// Multi-probe: all cells with L_∞ offset <= P, ordered by Σ o_i² then lexicographic o.
  /// Only non-empty cells are returned. `out_candidates` sums range lengths when non-null.
  std::vector<BucketRange> probe(std::span<const std::int32_t> query_bins,
                                 std::size_t probe_radius,
                                 std::size_t* out_candidates = nullptr) const;

 private:
  std::vector<std::uint64_t> keys_;
  std::vector<std::size_t> offsets_;  // size keys_+1
  ProjectionMatrix matrix_;
  BinCodec codec_;
};

}  // namespace vectorcache::index
