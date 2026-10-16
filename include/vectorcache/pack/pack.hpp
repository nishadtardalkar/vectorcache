#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "vectorcache/constants.hpp"

namespace vectorcache {

inline constexpr std::size_t kPerm0[16] = {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15};
inline constexpr std::size_t kInvPerm0[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};

struct BlockedGeometry {
  std::size_t n_blocks;
  std::size_t n_byte_groups;
  std::size_t blocked_len;
};

BlockedGeometry blocked_geometry(std::size_t n_vectors, std::size_t bits, std::size_t dim);

/// Runtime: AVX-512BW PERM0 when CPU supports it (TurboVec path), else AVX2 PERM0 / scalar.
bool use_avx512bw();

/// Runtime search kernel name: "avx512bw_perm0", "avx2_perm0", or "scalar".
std::string search_backend_name(std::size_t bits, std::size_t dim);

/// Multi-line dump of why a backend was chosen (CPUID, XCR0, env).
std::string search_backend_diagnostics(std::size_t bits, std::size_t dim);

/// Pack bit-plane codes into FastScan PERM0 layout (x86) or sequential (elsewhere).
std::pair<std::vector<std::uint8_t>, std::size_t> repack(std::span<const std::uint8_t> packed_codes,
                                                         std::size_t n_vectors, std::size_t bits,
                                                         std::size_t dim);

/// Alias of `repack` (PERM0-only; kept for call sites).
std::pair<std::vector<std::uint8_t>, std::size_t> repack_perm0(
    std::span<const std::uint8_t> packed_codes, std::size_t n_vectors, std::size_t bits,
    std::size_t dim);

/// Sequential blocked layout (arch-neutral).
std::vector<std::uint8_t> repack_seq(std::span<const std::uint8_t> packed_codes,
                                     std::size_t n_vectors, std::size_t bits, std::size_t dim);

/// Bit-plane packed → row-major flat codes (`n_vectors × n_byte_groups`) for sparse access.
std::vector<std::uint8_t> codes_flat_from_packed(std::span<const std::uint8_t> packed_codes,
                                                 std::size_t n_vectors, std::size_t bits,
                                                 std::size_t dim);

std::uint8_t read_code(std::span<const std::uint8_t> blocked, std::size_t bits,
                       std::size_t n_byte_groups, std::size_t block_idx, std::size_t g,
                       std::size_t lane);

}  // namespace vectorcache
