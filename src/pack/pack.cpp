#include "vectorcache/pack/pack.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

namespace vectorcache {
namespace {

#if defined(__x86_64__) || defined(_M_X64)
bool cpu_has_avx512_vnni_vbmi() {
#if defined(_MSC_VER)
  int info[4] = {};
  __cpuid(info, 0);
  if (info[0] < 7) {
    return false;
  }
  __cpuidex(info, 1, 0);
  if ((info[2] & (1 << 27)) == 0) {
    return false;  // OSXSAVE
  }
  const unsigned long long xcr0 = _xgetbv(0);
  // XCR0 bits 1,2,5,6,7: XMM, YMM, opmask, ZMM_hi256, ZMM_hi16
  if ((xcr0 & 0xE6ull) != 0xE6ull) {
    return false;
  }
  __cpuidex(info, 7, 0);
  const bool avx512f = (info[1] & (1 << 16)) != 0;
  const bool avx512bw = (info[1] & (1 << 30)) != 0;
  const bool avx512vbmi = (info[2] & (1 << 1)) != 0;
  const bool avx512vnni = (info[2] & (1 << 11)) != 0;
  return avx512f && avx512bw && avx512vbmi && avx512vnni;
#else
  unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
    return false;
  }
  if ((ecx & (1u << 27)) == 0) {
    return false;  // OSXSAVE
  }
  const unsigned long long xcr0 = _xgetbv(0);
  if ((xcr0 & 0xE6ull) != 0xE6ull) {
    return false;
  }
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
    return false;
  }
  const bool avx512f = (ebx & (1u << 16)) != 0;
  const bool avx512bw = (ebx & (1u << 30)) != 0;
  const bool avx512vbmi = (ecx & (1u << 1)) != 0;
  const bool avx512vnni = (ecx & (1u << 11)) != 0;
  return avx512f && avx512bw && avx512vbmi && avx512vnni;
#endif
}
#endif

std::array<std::array<std::uint32_t, 256>, 4> build_extract_lut(std::size_t bits) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t field = (bits == 3) ? 4 : bits;
  std::array<std::array<std::uint32_t, 256>, 4> lut{};
  for (std::size_t p = 0; p < bits; ++p) {
    for (std::size_t b = 0; b < 256; ++b) {
      std::uint32_t acc = 0;
      for (std::size_t j = 0; j < 8; ++j) {
        if (b & (1u << (7 - j))) {
          const std::size_t out_byte = j / codes_per_byte;
          const std::size_t shift_in_byte =
              (codes_per_byte - 1 - (j % codes_per_byte)) * field;
          acc |= 1u << (out_byte * 8 + shift_in_byte + p);
        }
      }
      lut[p][b] = acc;
    }
  }
  return lut;
}

const std::array<std::array<std::uint32_t, 256>, 4>& extract_lut(std::size_t bits) {
  static std::array<std::array<std::array<std::uint32_t, 256>, 4>, 5> tables;
  static std::once_flag flags[5];
  std::call_once(flags[bits], [&] { tables[bits] = build_extract_lut(bits); });
  return tables[bits];
}

std::vector<std::uint8_t> extract_codes_flat(std::span<const std::uint8_t> packed_codes,
                                            std::size_t n_vectors, std::size_t bits,
                                            std::size_t dim) {
  const std::size_t bytes_per_plane = dim / 8;
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  const std::size_t bytes_per_row = bits * bytes_per_plane;
  const std::size_t n_out = 8 / codes_per_byte;
  const auto& lut = extract_lut(bits);
  std::vector<std::uint8_t> codes_flat(n_vectors * n_byte_groups);
  for (std::size_t vec_idx = 0; vec_idx < n_vectors; ++vec_idx) {
    const std::size_t base = vec_idx * bytes_per_row;
    auto* row = codes_flat.data() + vec_idx * n_byte_groups;
    for (std::size_t c = 0; c < bytes_per_plane; ++c) {
      std::uint32_t acc = 0;
      for (std::size_t p = 0; p < bits; ++p) {
        acc |= lut[p][packed_codes[base + p * bytes_per_plane + c]];
      }
      // Little-endian bytes (matches turbovec `to_le_bytes`).
      std::uint8_t le[4] = {
          static_cast<std::uint8_t>(acc & 0xff),
          static_cast<std::uint8_t>((acc >> 8) & 0xff),
          static_cast<std::uint8_t>((acc >> 16) & 0xff),
          static_cast<std::uint8_t>((acc >> 24) & 0xff),
      };
      std::memcpy(row + c * n_out, le, n_out);
    }
  }
  return codes_flat;
}

std::vector<std::uint8_t> pack_blocked_sequential(std::size_t n, std::size_t n_blocks,
                                                 std::size_t n_byte_groups, std::size_t blocked_size,
                                                 std::span<const std::uint8_t> codes_flat) {
  std::vector<std::uint8_t> blocked(blocked_size, 0);
  for (std::size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const std::size_t base_vec = block_idx * kBlock;
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      const std::size_t out_offset = (block_idx * n_byte_groups + g) * kBlock;
      for (std::size_t lane = 0; lane < kBlock; ++lane) {
        const std::size_t vi = base_vec + lane;
        if (vi < n) {
          blocked[out_offset + lane] = codes_flat[vi * n_byte_groups + g];
        }
      }
    }
  }
  return blocked;
}

#if defined(__x86_64__) || defined(_M_X64)
std::vector<std::uint8_t> pack_blocked_perm0(std::size_t n, std::size_t n_blocks,
                                             std::size_t n_byte_groups, std::size_t blocked_size,
                                             std::span<const std::uint8_t> codes_flat) {
  // 192×1: PERM0 only defined on 32-lane tiles — pack each kFlushChunk sub-tile.
  std::vector<std::uint8_t> blocked(blocked_size, 0);
  for (std::size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const std::size_t base_vec = block_idx * kBlock;
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      const std::size_t out_offset = (block_idx * n_byte_groups + g) * kBlock;
      for (std::size_t chunk = 0; chunk < kChunksPerBlock; ++chunk) {
        const std::size_t chunk_base = base_vec + chunk * kFlushChunk;
        const std::size_t chunk_out = out_offset + chunk * kFlushChunk;
        for (std::size_t j = 0; j < 16; ++j) {
          const std::size_t va = chunk_base + kPerm0[j];
          const std::size_t vb = chunk_base + kPerm0[j] + 16;
          const std::uint8_t ba = (va < n) ? codes_flat[va * n_byte_groups + g] : 0;
          const std::uint8_t bb = (vb < n) ? codes_flat[vb * n_byte_groups + g] : 0;
          blocked[chunk_out + j] = static_cast<std::uint8_t>((ba >> 4) | ((bb >> 4) << 4));
          blocked[chunk_out + 16 + j] =
              static_cast<std::uint8_t>((ba & 0x0F) | ((bb & 0x0F) << 4));
        }
      }
    }
  }
  return blocked;
}
#endif

std::uint8_t deinterleave_x86_code_byte(std::span<const std::uint8_t> blocked, std::size_t group_off,
                                        std::size_t lane) {
  const std::size_t j = kInvPerm0[lane & 15];
  const std::uint8_t hi_plane = blocked[group_off + j];
  const std::uint8_t lo_plane = blocked[group_off + 16 + j];
  std::uint8_t hi, lo;
  if (lane < 16) {
    hi = hi_plane & 0x0F;
    lo = lo_plane & 0x0F;
  } else {
    hi = hi_plane >> 4;
    lo = lo_plane >> 4;
  }
  return static_cast<std::uint8_t>((hi << 4) | lo);
}

}  // namespace

BlockedGeometry blocked_geometry(std::size_t n_vectors, std::size_t bits, std::size_t dim) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  const std::size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
  return {n_blocks, n_byte_groups, n_blocks * n_byte_groups * kBlock};
}

bool use_vector_major() {
  if (const char* v = std::getenv("TURBOVEC_NO_VECTOR_MAJOR"); v && v[0] != '0') {
    return false;
  }
  if (const char* v = std::getenv("TURBOVEC_NO_VNNI"); v && v[0] != '0') {
    return false;
  }
  static int cpu_ok = -1;
  if (cpu_ok >= 0) {
    return cpu_ok != 0;
  }
#if defined(__x86_64__) || defined(_M_X64)
  // Prefer CPUID over __builtin_cpu_supports: GCC often reports false for
  // AVX-512 VNNI/VBMI when the TU is compiled with only -mavx2.
  cpu_ok = cpu_has_avx512_vnni_vbmi() ? 1 : 0;
#else
  cpu_ok = 0;
#endif
  return cpu_ok != 0;
}

bool vector_major_for([[maybe_unused]] std::size_t bits, std::size_t n_byte_groups) {
  // 192×1 layout: no %4 quad constraint (depth-1 byte-groups).
  (void)n_byte_groups;
#if defined(__x86_64__) || defined(_M_X64)
  const bool kernel_exists = true;
#else
  const bool kernel_exists = (bits == 4);
#endif
  return kernel_exists && use_vector_major();
}

std::string search_backend_name(std::size_t bits, std::size_t dim) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  if (vector_major_for(bits, n_byte_groups)) {
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
    return "avx512_vnni";
#else
    return (bits == 4) ? "avx512_permute_dot" : "avx512_vnni";
#endif
  }
#if defined(__x86_64__) || defined(_M_X64)
#if defined(__GNUC__) || defined(__clang__)
  if (__builtin_cpu_supports("avx2")) {
    return "avx2_perm0";
  }
#elif defined(_MSC_VER)
  int info[4] = {};
  __cpuid(info, 0);
  if (info[0] >= 7) {
    __cpuidex(info, 7, 0);
    if ((info[1] & (1 << 5)) != 0) {
      return "avx2_perm0";
    }
  }
#endif
#endif
  return "scalar";
}

std::string search_backend_diagnostics(std::size_t bits, std::size_t dim, std::size_t n_vectors) {
  std::ostringstream out;
  const std::size_t codes_per_byte = (bits == 0) ? 0 : (8 / bits);
  const std::size_t n_byte_groups_4 = (codes_per_byte == 0) ? 0 : (dim / codes_per_byte);
  const std::size_t n_byte_groups_sign = (dim % 8 == 0) ? (dim / 8) : 0;
  const bool vm = use_vector_major();
  const bool planes_cap = planes_for(bits, n_byte_groups_4);
  const bool planes_on =
      n_vectors > 0 ? planes_wanted(bits, n_byte_groups_4, n_vectors) : planes_cap;
  const bool sign_vm32 = planes_cap && n_byte_groups_sign > 0 && (n_byte_groups_sign % 4 == 0);

  out << "scan bits=" << bits << " dim=" << dim << " backend=" << search_backend_name(bits, dim)
      << " vm=" << (vm ? 1 : 0) << " planes=" << (planes_on ? 1 : 0)
      << " planes_min=" << planes_min_vectors() << "\n";
  out << "4bit groups=" << n_byte_groups_4 << " block=" << kBlock << " layout=192x1"
      << " kernel=" << (vm ? "permute_dot" : search_backend_name(bits, dim)) << "\n";
  if (planes_cap) {
    out << "sign groups=" << n_byte_groups_sign << " block=" << kSignBlock
        << " layout=" << (sign_vm32 ? "32x4" : "192x1")
        << " kernel=" << (sign_vm32 ? "vnni_vpermb" : "vnni_shuffle")
        << " nq=" << kNqBatchSign;
    if (n_vectors > 0 && !planes_on) {
      out << " (inactive: n < planes_min)";
    }
    out << "\n";
  }

  // Only print kill switches when they disable a path.
  if (const char* v = std::getenv("TURBOVEC_4BIT_PLANES"); v != nullptr && v[0] == '0') {
    out << "TURBOVEC_4BIT_PLANES=0 (planes off)\n";
  }
  if (const char* v = std::getenv("TURBOVEC_NO_VECTOR_MAJOR"); v != nullptr && v[0] != '\0' &&
                                                              v[0] != '0') {
    out << "TURBOVEC_NO_VECTOR_MAJOR=" << v << "\n";
  }
  if (const char* v = std::getenv("TURBOVEC_NO_VNNI"); v != nullptr && v[0] != '\0' && v[0] != '0') {
    out << "TURBOVEC_NO_VNNI=" << v << "\n";
  }

#if defined(__x86_64__) || defined(_M_X64)
  const bool isa_ok = cpu_has_avx512_vnni_vbmi();
  out << "isa avx512_vbmi_vnni=" << (isa_ok ? 1 : 0);
  if (!isa_ok && !vm) {
    out << " (fallback " << search_backend_name(bits, dim) << ")";
  }
  out << "\n";
#else
  out << "isa=non-x86_64\n";
#endif
  return out.str();
}

void vector_major_chunk(std::span<std::uint8_t> buf) {
  // Full 4-bit codes stay 192×1 (identity). Sign-plane 32×4 uses vector_major_sign_chunk.
  (void)buf;
}

void vector_major_to_seq_chunk(std::span<std::uint8_t> buf) {
  (void)buf;
}

/// Sequential blocked (BLOCK=32) → turbovec vector-major: 4 groups × 32 vectors per unit.
void vector_major_sign_chunk(std::span<std::uint8_t> buf) {
  if (buf.size() % kSignVmUnit != 0) {
    return;
  }
  alignas(64) std::uint8_t tmp[kSignVmUnit];
  for (std::size_t off = 0; off < buf.size(); off += kSignVmUnit) {
    std::memcpy(tmp, buf.data() + off, kSignVmUnit);
    for (std::size_t j = 0; j < 4; ++j) {
      for (std::size_t v = 0; v < kSignBlock; ++v) {
        buf[off + (v / 16) * 64 + (v % 16) * 4 + j] = tmp[j * kSignBlock + v];
      }
    }
  }
}

std::vector<std::uint8_t> codes_flat_from_packed(std::span<const std::uint8_t> packed_codes,
                                                 std::size_t n_vectors, std::size_t bits,
                                                 std::size_t dim) {
  return extract_codes_flat(packed_codes, n_vectors, bits, dim);
}

std::vector<std::uint8_t> repack_seq(std::span<const std::uint8_t> packed_codes,
                                     std::size_t n_vectors, std::size_t bits, std::size_t dim) {
  const auto geo = blocked_geometry(n_vectors, bits, dim);
  auto codes_flat = extract_codes_flat(packed_codes, n_vectors, bits, dim);
  return pack_blocked_sequential(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                 codes_flat);
}

std::pair<std::vector<std::uint8_t>, std::size_t> repack(std::span<const std::uint8_t> packed_codes,
                                                         std::size_t n_vectors, std::size_t bits,
                                                         std::size_t dim) {
  const auto geo = blocked_geometry(n_vectors, bits, dim);
  auto codes_flat = extract_codes_flat(packed_codes, n_vectors, bits, dim);
  std::vector<std::uint8_t> blocked;
  if (vector_major_for(bits, geo.n_byte_groups)) {
    blocked = pack_blocked_sequential(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                      codes_flat);
    vector_major_chunk(blocked);
  } else {
#if defined(__x86_64__) || defined(_M_X64)
    blocked = pack_blocked_perm0(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                 codes_flat);
#else
    blocked = pack_blocked_sequential(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                      codes_flat);
#endif
  }
  return {std::move(blocked), geo.n_blocks};
}

std::pair<std::vector<std::uint8_t>, std::size_t> repack_perm0(
    std::span<const std::uint8_t> packed_codes, std::size_t n_vectors, std::size_t bits,
    std::size_t dim) {
  const auto geo = blocked_geometry(n_vectors, bits, dim);
  auto codes_flat = extract_codes_flat(packed_codes, n_vectors, bits, dim);
#if defined(__x86_64__) || defined(_M_X64)
  auto blocked = pack_blocked_perm0(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                    codes_flat);
#else
  auto blocked = pack_blocked_sequential(n_vectors, geo.n_blocks, geo.n_byte_groups, geo.blocked_len,
                                         codes_flat);
#endif
  return {std::move(blocked), geo.n_blocks};
}

std::uint8_t read_code(std::span<const std::uint8_t> blocked, std::size_t bits,
                       std::size_t n_byte_groups, std::size_t block_idx, std::size_t g,
                       std::size_t lane) {
  const std::size_t block_base = block_idx * n_byte_groups * kBlock;
  const std::size_t group_off = block_base + g * kBlock;
  if (vector_major_for(bits, n_byte_groups)) {
    // 192×1 sequential: one byte-group is kBlock contiguous vector bytes.
    return blocked[group_off + lane];
  }
#if defined(__x86_64__) || defined(_M_X64)
  // PERM0 is defined on kFlushChunk tiles within the 192-wide group.
  const std::size_t chunk = lane / kFlushChunk;
  const std::size_t lane_in_chunk = lane % kFlushChunk;
  return deinterleave_x86_code_byte(blocked, group_off + chunk * kFlushChunk, lane_in_chunk);
#else
  return blocked[group_off + lane];
#endif
}

bool use_planes4() {
  if (const char* v = std::getenv("TURBOVEC_4BIT_PLANES"); v != nullptr && v[0] == '0') {
    return false;  // "0", "0…": kill switch
  }
  return true;
}

std::size_t planes_min_vectors() {
  if (const char* v = std::getenv("VECTORCACHE_PLANES_MIN"); v && v[0] != '\0') {
    char* end = nullptr;
    const unsigned long parsed = std::strtoul(v, &end, 10);
    if (end != v && parsed > 0) {
      return static_cast<std::size_t>(parsed);
    }
  }
  return 32768;
}

bool planes_for(std::size_t bits, std::size_t n_byte_groups) {
  // 4-bit staged: sign region nsg = dim/8; keep dim % 8 == 0 via byte packing.
  // Former %16 (32×4) gate dropped for 192×1.
  (void)n_byte_groups;
  return bits == 4 && use_planes4() && use_vector_major();
}

bool planes_wanted(std::size_t bits, std::size_t n_byte_groups, std::size_t n_vectors) {
  return n_vectors > 0 && n_vectors >= planes_min_vectors() && planes_for(bits, n_byte_groups);
}

std::size_t planes_shortlist_len(std::size_t k, std::size_t bits) {
  if (bits == 4) {
    return std::max<std::size_t>(256, (k >= 64) ? (k * 16) : (k * 20));
  }
  return std::max<std::size_t>(384, k * 16);
}

std::size_t planes_exact_len(std::size_t k, std::size_t shortlist_len) {
  // Without mid-plane ranking, exact the turbovec mid cut (max(256, 6k)) so we
  // don't densify/rescan the full sign shortlist (16k). Still ≫ final k.
  const std::size_t want = std::max<std::size_t>(256, 6 * k);
  return std::min(shortlist_len, want);
}

std::pair<std::vector<std::uint8_t>, std::size_t> pack_flat_for_search(
    std::span<const std::uint8_t> codes_flat, std::size_t n_vectors, std::size_t bits,
    std::size_t n_byte_groups) {
  if (n_vectors == 0 || n_byte_groups == 0) {
    return {{}, 0};
  }
  if (codes_flat.size() < n_vectors * n_byte_groups) {
    return {{}, 0};
  }
  const std::size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
  const std::size_t blocked_len = n_blocks * n_byte_groups * kBlock;
  std::vector<std::uint8_t> blocked;
  if (vector_major_for(bits, n_byte_groups)) {
    blocked = pack_blocked_sequential(n_vectors, n_blocks, n_byte_groups, blocked_len, codes_flat);
    vector_major_chunk(blocked);
  } else {
#if defined(__x86_64__) || defined(_M_X64)
    blocked = pack_blocked_perm0(n_vectors, n_blocks, n_byte_groups, blocked_len, codes_flat);
#else
    blocked = pack_blocked_sequential(n_vectors, n_blocks, n_byte_groups, blocked_len, codes_flat);
#endif
  }
  return {std::move(blocked), n_blocks};
}

std::pair<std::vector<std::uint8_t>, std::size_t> planes_repack_sign(
    std::span<const std::uint8_t> packed_codes, std::size_t n_vectors, std::size_t bits,
    std::size_t dim) {
  if (bits < 2 || bits > 4 || dim == 0 || dim % 8 != 0 || n_vectors == 0) {
    return {{}, 0};
  }
  const std::size_t nsg = dim / 8;
  const std::size_t bytes_per_row = bits * nsg;
  const std::size_t sign_off = (bits - 1) * nsg;
  std::vector<std::uint8_t> sign_rows(n_vectors * nsg);
  for (std::size_t v = 0; v < n_vectors; ++v) {
    const std::uint8_t* row = packed_codes.data() + v * bytes_per_row + sign_off;
    std::memcpy(sign_rows.data() + v * nsg, row, nsg);
  }
  // True VNNI (32×4 VM) when groups are a multiple of 4 — same gate as turbovec.
  const bool use_vm32 = vector_major_for(/*bits=*/2, nsg) && (nsg % 4 == 0);
  if (use_vm32) {
    const std::size_t n_blocks = (n_vectors + kSignBlock - 1) / kSignBlock;
    const std::size_t blocked_len = n_blocks * nsg * kSignBlock;
    std::vector<std::uint8_t> blocked(blocked_len, 0);
    for (std::size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
      const std::size_t base_vec = block_idx * kSignBlock;
      for (std::size_t g = 0; g < nsg; ++g) {
        const std::size_t out_offset = (block_idx * nsg + g) * kSignBlock;
        for (std::size_t lane = 0; lane < kSignBlock; ++lane) {
          const std::size_t vi = base_vec + lane;
          if (vi < n_vectors) {
            blocked[out_offset + lane] = sign_rows[vi * nsg + g];
          }
        }
      }
    }
    vector_major_sign_chunk(blocked);
    return {std::move(blocked), n_blocks};
  }
  // Fallback: 192×1 (shuffle LUT / AVX2 PERM0).
  const std::size_t n_blocks = (n_vectors + kBlock - 1) / kBlock;
  const std::size_t blocked_len = n_blocks * nsg * kBlock;
  std::vector<std::uint8_t> blocked;
  if (vector_major_for(/*bits=*/2, nsg)) {
    blocked = pack_blocked_sequential(n_vectors, n_blocks, nsg, blocked_len, sign_rows);
  } else {
#if defined(__x86_64__) || defined(_M_X64)
    blocked = pack_blocked_perm0(n_vectors, n_blocks, nsg, blocked_len, sign_rows);
#else
    blocked = pack_blocked_sequential(n_vectors, n_blocks, nsg, blocked_len, sign_rows);
#endif
  }
  return {std::move(blocked), n_blocks};
}

}  // namespace vectorcache
