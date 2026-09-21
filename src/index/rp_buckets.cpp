#include "vectorcache/index/rp_buckets.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <random>
#include <vector>

#include "vectorcache/error.hpp"

namespace vectorcache::index {
namespace {

constexpr float kNormEps = 1e-12f;

}  // namespace

PairHash::PairHash(std::size_t num_pair_dirs, std::uint64_t seed) : num_pair_dirs_(num_pair_dirs) {
  if (num_pair_dirs_ == 0 || num_pair_dirs_ > kMaxPairDirs) {
    throw Error("PairHash: num_pair_dirs must be in 1..kMaxPairDirs");
  }
  dirs_.assign(num_pair_dirs_ * 2, 0.0f);
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> gauss(0.0f, 1.0f);
  for (std::size_t i = 0; i < num_pair_dirs_; ++i) {
    float r0 = gauss(rng);
    float r1 = gauss(rng);
    const double energy =
        static_cast<double>(r0) * static_cast<double>(r0) +
        static_cast<double>(r1) * static_cast<double>(r1);
    if (energy <= 0.0) {
      r0 = 1.0f;
      r1 = 0.0f;
    } else {
      const float inv = static_cast<float>(1.0 / std::sqrt(energy));
      r0 *= inv;
      r1 *= inv;
    }
    dirs_[2 * i] = r0;
    dirs_[2 * i + 1] = r1;
  }
}

std::span<const float> PairHash::dir(std::size_t i) const {
  if (i >= num_pair_dirs_) {
    throw Error("PairHash::dir out of range");
  }
  return {dirs_.data() + 2 * i, 2};
}

float PairHash::fold(std::span<const float> x) const {
  if (empty()) {
    throw Error("PairHash::fold: empty hash");
  }
  if (x.empty()) {
    throw Error("PairHash::fold: empty input");
  }

  std::vector<float> cur(x.begin(), x.end());
  std::vector<float> next;
  next.reserve((cur.size() + 1) / 2);
  std::size_t cursor = 0;

  while (cur.size() > 1) {
    next.clear();
    const std::size_t n = cur.size();
    const std::size_t pairs = n / 2;
    for (std::size_t i = 0; i < pairs; ++i) {
      const float a = cur[2 * i];
      const float b = cur[2 * i + 1];
      const float n2 = a * a + b * b;
      float out = 0.0f;
      if (n2 > kNormEps) {
        const float inv = 1.0f / std::sqrt(n2);
        const float p0 = a * inv;
        const float p1 = b * inv;
        const std::size_t di = cursor % num_pair_dirs_;
        out = p0 * dirs_[2 * di] + p1 * dirs_[2 * di + 1];
      }
      ++cursor;
      next.push_back(out);
    }
    if ((n & 1u) != 0) {
      next.push_back(cur[n - 1]);
    }
    cur.swap(next);
  }
  return cur[0];
}

float arcsine_cdf(float s) {
  const float clamped_s = std::clamp(s, -1.0f, 1.0f);
  const float u =
      0.5f + std::asin(clamped_s) / static_cast<float>(std::numbers::pi);
  return std::clamp(u, 0.0f, 1.0f);
}

std::int32_t fold_to_bin(const PairHash& hash, std::span<const float> x, float bin_width) {
  if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
    throw Error("bin_width must be finite and > 0");
  }
  const float u = arcsine_cdf(hash.fold(x));
  return static_cast<std::int32_t>(std::floor(static_cast<double>(u) / bin_width));
}

std::uint64_t pack_bin(std::int32_t bin) {
  return static_cast<std::uint64_t>(static_cast<std::int64_t>(bin));
}

void validate_probe_radius(std::size_t probe_radius) {
  const std::size_t cells = 2 * probe_radius + 1;
  if (cells > kMaxProbeCells) {
    throw Error("probe radius: 2P+1 exceeds kMaxProbeCells");
  }
}

BucketIndex BucketIndex::build(std::span<const std::uint64_t> sorted_keys, PairHash hash,
                               float bin_width) {
  if (sorted_keys.empty()) {
    throw Error("BucketIndex::build: empty keys");
  }
  if (hash.empty()) {
    throw Error("BucketIndex::build: empty PairHash");
  }
  if (bin_width <= 0.0f || !std::isfinite(bin_width)) {
    throw Error("BucketIndex::build: bin_width must be finite and > 0");
  }
  for (std::size_t i = 1; i < sorted_keys.size(); ++i) {
    if (sorted_keys[i] < sorted_keys[i - 1]) {
      throw Error("BucketIndex::build: keys must be sorted ascending");
    }
  }

  BucketIndex idx;
  idx.hash_ = std::move(hash);
  idx.bin_width_ = bin_width;
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

BucketRange BucketIndex::cell(std::size_t i) const {
  if (i >= keys_.size()) {
    throw Error("BucketIndex::cell: index out of range");
  }
  const std::size_t start = offsets_[i];
  const std::size_t end = offsets_[i + 1];
  return {start, end - start};
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

std::vector<BucketRange> BucketIndex::probe(std::int32_t query_bin, std::size_t probe_radius,
                                            std::size_t* out_candidates) const {
  if (empty()) {
    throw Error("BucketIndex::probe: empty index");
  }
  validate_probe_radius(probe_radius);

  // Order by |offset| then signed offset: 0, -1, +1, -2, +2, ...
  std::vector<std::int32_t> offsets;
  offsets.reserve(2 * probe_radius + 1);
  offsets.push_back(0);
  for (std::size_t d = 1; d <= probe_radius; ++d) {
    offsets.push_back(-static_cast<std::int32_t>(d));
    offsets.push_back(static_cast<std::int32_t>(d));
  }

  std::vector<BucketRange> ranges;
  ranges.reserve(offsets.size());
  std::size_t candidates = 0;
  for (const std::int32_t o : offsets) {
    const std::int32_t bin = query_bin + o;
    const BucketRange range = find(pack_bin(bin));
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
