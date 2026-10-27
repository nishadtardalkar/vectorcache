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
/// Full 4-bit codes stay 192×1 (identity). Sign plane uses `kSignVmUnit` (32×4).
inline constexpr std::size_t kVmUnit = kBlock;

struct BlockedGeometry {
  std::size_t n_blocks;
  std::size_t n_byte_groups;
  std::size_t blocked_len;
};

BlockedGeometry blocked_geometry(std::size_t n_vectors, std::size_t bits, std::size_t dim);

bool use_vector_major();
bool vector_major_for(std::size_t bits, std::size_t n_byte_groups);

/// Runtime search kernel name: "avx512_permute_dot", "avx512_vnni",
/// "avx2_perm0", or "scalar".
std::string search_backend_name(std::size_t bits, std::size_t dim);

/// Compact dump: scan geometry, sign vs 4-bit layouts, planes, ISA.
/// Pass `n_vectors` so `planes_active` reflects the size gate (not just ISA).
std::string search_backend_diagnostics(std::size_t bits, std::size_t dim,
                                       std::size_t n_vectors = 0);

/// Pack bit-plane codes into native search layout (x86: PERM0 or vector-major).
std::pair<std::vector<std::uint8_t>, std::size_t> repack(std::span<const std::uint8_t> packed_codes,
                                                         std::size_t n_vectors, std::size_t bits,
                                                         std::size_t dim);

/// Always pack AVX2 PERM0 layout (for landmark FastScan paired with avx2_perm0).
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

void vector_major_chunk(std::span<std::uint8_t> buf);
void vector_major_to_seq_chunk(std::span<std::uint8_t> buf);

/// turbovec planes: 4-bit staged search (sign plane → shortlist → exact).
/// Kill switch: `TURBOVEC_4BIT_PLANES=0`. Optional `VECTORCACHE_PLANES_MIN` (default 32768).
bool use_planes4();
std::size_t planes_min_vectors();
/// Geometry gate: `bits==4`, VNNI host (192×1; dim multiple of 8 from packing).
bool planes_for(std::size_t bits, std::size_t n_byte_groups);
bool planes_wanted(std::size_t bits, std::size_t n_byte_groups, std::size_t n_vectors);

/// Shortlist length after the sign-plane pass (turbovec `planes_shortlist_len`).
std::size_t planes_shortlist_len(std::size_t k, std::size_t bits);

/// Extract MSB bit-plane from packed rows and FastScan-pack it as a 2-bit-shaped
/// sign region (`n_byte_groups = dim/8`). On VNNI hosts with `nsg % 4 == 0`, packs
/// turbovec 32×4 vector-major for `vpermb`+`vpdpbusd`; else 192×1. Returns
/// (sign_blocked, n_blocks) with block width matching the layout.
std::pair<std::vector<std::uint8_t>, std::size_t> planes_repack_sign(
    std::span<const std::uint8_t> packed_codes, std::size_t n_vectors, std::size_t bits,
    std::size_t dim);

/// Pack row-major flat codes (`n × n_byte_groups`) into the native search layout
/// (vector-major when eligible, else PERM0). Used to densify a shortlist for exact rescore.
std::pair<std::vector<std::uint8_t>, std::size_t> pack_flat_for_search(
    std::span<const std::uint8_t> codes_flat, std::size_t n_vectors, std::size_t bits,
    std::size_t n_byte_groups);

/// How many sign-shortlist hits get the exact 4-bit rescore (turbovec mid/rescore floor).
std::size_t planes_exact_len(std::size_t k, std::size_t shortlist_len);

}  // namespace vectorcache
