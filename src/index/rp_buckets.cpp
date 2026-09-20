#include "vectorcache/index/rp_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <string>

#include "vectorcache/error.hpp"
#include "vectorcache/simd.hpp"

namespace vectorcache::index {
namespace {

float dot_avx(std::span<const float> a, std::span<const float> b) {
  const std::size_t n = a.size();
  std::size_t i = 0;
  __m512 acc = _mm512_setzero_ps();
  // Row starts are 64B-aligned when dim is a multiple of 16 (AlignedVector base).
  const bool a_aligned =
      (reinterpret_cast<std::uintptr_t>(a.data()) & 63u) == 0;
  for (; i + simd::kWidth <= n; i += simd::kWidth) {
    const __m512 va =
        a_aligned ? _mm512_load_ps(a.data() + i) : _mm512_loadu_ps(a.data() + i);
    const __m512 vb = _mm512_loadu_ps(b.data() + i);
    acc = _mm512_fmadd_ps(va, vb, acc);
  }
  float sum = _mm512_reduce_add_ps(acc);
  for (; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

struct ProbeOffset {
  std::int32_t o[kMaxProjections];
  std::size_t r = 0;
  int sum_sq = 0;
};

void enumerate_offsets(std::size_t R, std::size_t P, std::vector<ProbeOffset>& out) {
  out.clear();
  const int p = static_cast<int>(P);
  const int side = 2 * p + 1;
  std::size_t total = 1;
  for (std::size_t i = 0; i < R; ++i) {
    total *= static_cast<std::size_t>(side);
  }
  out.reserve(total);

  std::vector<int> cur(R, -p);
  for (;;) {
    ProbeOffset po;
    po.r = R;
    po.sum_sq = 0;
    for (std::size_t i = 0; i < R; ++i) {
      po.o[i] = static_cast<std::int32_t>(cur[i]);
      po.sum_sq += cur[i] * cur[i];
    }
    out.push_back(po);

    std::size_t axis = 0;
    while (axis < R) {
      ++cur[axis];
      if (cur[axis] <= p) {
        break;
      }
      cur[axis] = -p;
      ++axis;
    }
    if (axis == R) {
      break;
    }
  }

  std::stable_sort(out.begin(), out.end(), [](const ProbeOffset& a, const ProbeOffset& b) {
    if (a.sum_sq != b.sum_sq) {
      return a.sum_sq < b.sum_sq;
    }
    for (std::size_t i = 0; i < a.r; ++i) {
      if (a.o[i] != b.o[i]) {
        return a.o[i] < b.o[i];
      }
    }
    return false;
  });
}

const std::vector<ProbeOffset>& cached_probe_offsets(std::size_t R, std::size_t P) {
  struct Entry {
    std::size_t R = 0;
    std::size_t P = 0;
    std::vector<ProbeOffset> offsets;
  };
  static Entry cache;
  if (cache.R == R && cache.P == P && !cache.offsets.empty()) {
    return cache.offsets;
  }
  cache.R = R;
  cache.P = P;
  enumerate_offsets(R, P, cache.offsets);
  return cache.offsets;
}

}  // namespace

ProjectionMatrix::ProjectionMatrix(std::size_t num_projections, std::size_t dim,
                                   std::uint64_t seed)
    : num_projections_(num_projections), dim_(dim) {
  if (num_projections_ == 0 || dim_ == 0) {
    throw Error("ProjectionMatrix: num_projections and dim must be > 0");
  }
  if (num_projections_ > kMaxProjections) {
    throw Error("ProjectionMatrix: num_projections exceeds kMaxProjections");
  }
  data_.assign(num_projections_ * dim_, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  for (std::size_t r = 0; r < num_projections_; ++r) {
    float* row = data_.data() + r * dim_;
    double energy = 0.0;
    for (std::size_t j = 0; j < dim_; ++j) {
      row[j] = gauss(rng);
      energy += static_cast<double>(row[j]) * static_cast<double>(row[j]);
    }
    if (energy <= 0.0) {
      row[0] = 1.0f;
      energy = 1.0;
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(energy));
    for (std::size_t j = 0; j < dim_; ++j) {
      row[j] *= inv;
    }
  }
}

std::span<const float> ProjectionMatrix::row(std::size_t r) const {
  if (r >= num_projections_) {
    throw Error("ProjectionMatrix::row out of range");
  }
  return {data_.data() + r * dim_, dim_};
}

void ProjectionMatrix::project(std::span<const float> x, std::span<float> out) const {
  if (x.size() != dim_) {
    throw Error("ProjectionMatrix::project dimension mismatch");
  }
  if (out.size() != num_projections_) {
    throw Error("ProjectionMatrix::project output size mismatch");
  }
  for (std::size_t r = 0; r < num_projections_; ++r) {
    out[r] = dot_avx(row(r), x);
  }
}

void project_to_bins(const ProjectionMatrix& matrix, std::span<const float> x, float bin_width,
                     std::span<std::int32_t> bins) {
  if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
    throw Error("bin_width must be finite and > 0");
  }
  if (bins.size() != matrix.num_projections()) {
    throw Error("project_to_bins: bins size mismatch");
  }
  alignas(64) float projs[kMaxProjections];
  matrix.project(x, std::span<float>(projs, matrix.num_projections()));
  for (std::size_t r = 0; r < matrix.num_projections(); ++r) {
    bins[r] = static_cast<std::int32_t>(std::floor(static_cast<double>(projs[r]) / bin_width));
  }
}

BinCodec make_bin_codec(std::size_t num_projections, float bin_width) {
  if (num_projections == 0 || num_projections > kMaxProjections) {
    throw Error("make_bin_codec: invalid num_projections");
  }
  if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
    throw Error("make_bin_codec: bin_width must be finite and > 0");
  }
  // Pack R signed bins into uint64. bits_per_axis = min(32, 64/R).
  const std::uint32_t bits =
      std::min<std::uint32_t>(32u, static_cast<std::uint32_t>(64u / num_projections));
  if (bits == 0 || static_cast<std::uint64_t>(bits) * num_projections > 64ull) {
    throw Error("make_bin_codec: cannot pack R-tuple into uint64");
  }
  // Signed range centered at 0: [-(2^(bits-1)), 2^(bits-1) - 1]
  const std::int32_t half = static_cast<std::int32_t>(1u << (bits - 1));
  BinCodec codec;
  codec.bin_width = bin_width;
  codec.bin_lo = -half;
  codec.bits_per_axis = bits;
  codec.num_projections = num_projections;
  return codec;
}

std::uint64_t pack_cell_key_unchecked(std::span<const std::int32_t> bins, const BinCodec& codec) {
  const std::uint64_t mask =
      codec.bits_per_axis == 64 ? ~0ull : ((1ull << codec.bits_per_axis) - 1ull);
  std::uint64_t key = 0;
  for (std::size_t i = 0; i < codec.num_projections; ++i) {
    const std::uint64_t shifted =
        static_cast<std::uint64_t>(static_cast<std::int64_t>(bins[i]) - codec.bin_lo);
    key = (key << codec.bits_per_axis) | (shifted & mask);
  }
  return key;
}

std::uint64_t pack_cell_key(std::span<const std::int32_t> bins, const BinCodec& codec) {
  if (bins.size() != codec.num_projections || !codec.packable()) {
    throw Error("pack_cell_key: invalid bins/codec");
  }
  const std::uint64_t mask =
      codec.bits_per_axis == 64 ? ~0ull : ((1ull << codec.bits_per_axis) - 1ull);
  std::uint64_t key = 0;
  for (std::size_t i = 0; i < codec.num_projections; ++i) {
    const std::int64_t shifted = static_cast<std::int64_t>(bins[i]) - codec.bin_lo;
    if (shifted < 0 || static_cast<std::uint64_t>(shifted) > mask) {
      throw Error("pack_cell_key: bin out of codec range");
    }
    key = (key << codec.bits_per_axis) | (static_cast<std::uint64_t>(shifted) & mask);
  }
  return key;
}

void unpack_cell_key(std::uint64_t key, const BinCodec& codec, std::span<std::int32_t> bins) {
  if (bins.size() != codec.num_projections || !codec.packable()) {
    throw Error("unpack_cell_key: invalid bins/codec");
  }
  const std::uint64_t mask =
      codec.bits_per_axis == 64 ? ~0ull : ((1ull << codec.bits_per_axis) - 1ull);
  for (std::size_t i = codec.num_projections; i > 0; --i) {
    const std::size_t idx = i - 1;
    bins[idx] = static_cast<std::int32_t>((key & mask) + static_cast<std::uint64_t>(codec.bin_lo));
    key >>= codec.bits_per_axis;
  }
}

void validate_probe_grid(std::size_t num_projections, std::size_t probe_radius) {
  if (num_projections == 0 || num_projections > kMaxProjections) {
    throw Error("validate_probe_grid: invalid num_projections");
  }
  std::size_t cells = 1;
  const std::size_t side = 2 * probe_radius + 1;
  for (std::size_t i = 0; i < num_projections; ++i) {
    if (cells > kMaxProbeCells / side) {
      throw Error("probe grid (2P+1)^R exceeds kMaxProbeCells");
    }
    cells *= side;
  }
  if (cells > kMaxProbeCells) {
    throw Error("probe grid (2P+1)^R exceeds kMaxProbeCells");
  }
}

BucketIndex BucketIndex::build(std::span<const std::uint64_t> sorted_keys, ProjectionMatrix matrix,
                               BinCodec codec) {
  if (sorted_keys.empty()) {
    throw Error("BucketIndex::build: empty keys");
  }
  if (matrix.num_projections() != codec.num_projections) {
    throw Error("BucketIndex::build: matrix/codec projection mismatch");
  }
  if (!codec.packable()) {
    throw Error("BucketIndex::build: codec not packable");
  }
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] < sorted_keys[i - 1]) {
      throw Error("BucketIndex::build: keys must be sorted ascending");
    }
  }

  BucketIndex idx;
  idx.matrix_ = std::move(matrix);
  idx.codec_ = codec;
  idx.keys_.reserve(sorted_keys.size());
  idx.offsets_.reserve(sorted_keys.size() + 1);
  idx.offsets_.push_back(0);

  std::uint64_t cur = sorted_keys[0];
  idx.keys_.push_back(cur);
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] != cur) {
      idx.offsets_.push_back(i);
      cur = sorted_keys[i];
      idx.keys_.push_back(cur);
    }
  }
  idx.offsets_.push_back(sorted_keys.size());
  return idx;
}

BucketRange BucketIndex::find(std::uint64_t key) const {
  const auto it = std::lower_bound(keys_.begin(), keys_.end(), key);
  if (it == keys_.end() || *it != key) {
    return {};
  }
  const std::size_t cell = static_cast<std::size_t>(it - keys_.begin());
  const std::size_t start = offsets_[cell];
  const std::size_t end = offsets_[cell + 1];
  return {start, end - start};
}

std::vector<BucketRange> BucketIndex::probe(std::span<const std::int32_t> query_bins,
                                            std::size_t probe_radius,
                                            std::size_t* out_candidates) const {
  if (empty()) {
    throw Error("BucketIndex::probe: empty index");
  }
  if (query_bins.size() != codec_.num_projections) {
    throw Error("BucketIndex::probe: query_bins size mismatch");
  }
  validate_probe_grid(codec_.num_projections, probe_radius);

  const std::vector<ProbeOffset>& offsets =
      cached_probe_offsets(codec_.num_projections, probe_radius);

  std::vector<BucketRange> ranges;
  ranges.reserve(offsets.size());
  std::size_t candidates = 0;
  alignas(64) std::int32_t bins[kMaxProjections];

  for (const auto& po : offsets) {
    for (std::size_t i = 0; i < codec_.num_projections; ++i) {
      bins[i] = query_bins[i] + po.o[i];
    }
    const std::uint64_t key = pack_cell_key_unchecked(
        std::span<const std::int32_t>(bins, codec_.num_projections), codec_);
    const BucketRange range = find(key);
    if (range.length == 0) {
      continue;
    }
    ranges.push_back(range);
    candidates += range.length;
  }
  if (out_candidates != nullptr) {
    *out_candidates = candidates;
  }
  return ranges;
}

}  // namespace vectorcache::index
