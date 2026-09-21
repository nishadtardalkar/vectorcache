#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include "vectorcache/aligned.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::query {

/// Exact float query LUT for byte-aligned TurboQuantMSE codes (bits 1/2/4/8).
/// One 256-entry table per packed byte-group of **codes** (not dims).
/// For bits=1, also stores AVX-512 mask-add deltas over codes:
/// score = base + sum_{bit b set} delta[b], with delta length = num_codes.
class QueryLut {
 public:
  static constexpr std::size_t kEntries = 256;

  QueryLut() = default;

  bool empty() const { return tables_.empty() && delta_.empty(); }
  std::size_t num_groups() const { return num_groups_; }
  std::size_t bits() const { return bits_; }
  std::size_t bytes_per_vector() const { return num_groups_; }
  std::size_t num_codes() const { return num_codes_; }

  /// Flat layout: table for group g starts at tables_[g * kEntries].
  const float* table(std::size_t group) const { return tables_.data() + group * kEntries; }
  float* table(std::size_t group) { return tables_.data() + group * kEntries; }

  bool has_bit1_deltas() const { return !delta_.empty(); }
  float bit1_base() const { return bit1_base_; }
  const float* bit1_delta() const { return delta_.data(); }

  void clear() {
    tables_.clear();
    delta_.clear();
    num_groups_ = 0;
    bits_ = 0;
    num_codes_ = 0;
    bit1_base_ = 0.0f;
  }

  void resize(std::size_t num_groups, std::size_t bits, std::size_t num_codes) {
    num_groups_ = num_groups;
    bits_ = bits;
    num_codes_ = num_codes;
    const std::size_t need = num_groups * kEntries;
    if (tables_.size() != need) {
      tables_.resize(need);  // caller fills every entry; skip zero-fill on reuse
    }
    delta_.clear();
    bit1_base_ = 0.0f;
  }

  void set_bit1_deltas(float base, AlignedVector<float> delta) {
    bit1_base_ = base;
    delta_ = std::move(delta);
  }

 private:
  AlignedVector<float> tables_;
  AlignedVector<float> delta_;
  std::size_t num_groups_ = 0;
  std::size_t bits_ = 0;
  std::size_t num_codes_ = 0;
  float bit1_base_ = 0.0f;
};

/// True when bits divides 8 (LUT path). Odd widths use scalar fallback.
bool lut_bits_supported(std::size_t bits);

/// Build exact float LUTs over packed block codes.
void build_query_lut(std::span<const float> query_rotated, const quantize::LloydMaxCodebook& codebook,
                     QueryLut& out);

/// Asymmetric inner-product score: sum_i q_rot[i] * xhat[i] for one DB vector.
float asymmetric_ip_score(std::span<const float> query_rotated,
                          std::span<const std::uint64_t> data_words,
                          const quantize::LloydMaxCodebook& codebook);

/// Score one packed row with a prebuilt LUT (must match codebook bits/dim).
float asymmetric_ip_score_lut(const QueryLut& lut, std::span<const std::uint64_t> data_words);

/// Score num_vectors contiguous packed codes against a rotated float query.
void asymmetric_ip_batch(std::span<const float> query_rotated,
                         std::span<const std::uint64_t> data_words, std::size_t data_words_per_vec,
                         std::size_t num_vectors, const quantize::LloydMaxCodebook& codebook,
                         std::span<float> out_scores);

/// Score with a prebuilt LUT (preferred hot path).
void asymmetric_ip_batch_lut(const QueryLut& lut, std::span<const std::uint64_t> data_words,
                             std::size_t data_words_per_vec, std::size_t num_vectors,
                             const quantize::LloydMaxCodebook& codebook,
                             std::span<const float> query_rotated, std::span<float> out_scores);

}  // namespace vectorcache::query
