#include "vectorcache/pack/pack.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <mutex>
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
  std::vector<std::uint8_t> blocked(blocked_size, 0);
  for (std::size_t block_idx = 0; block_idx < n_blocks; ++block_idx) {
    const std::size_t base_vec = block_idx * kBlock;
    for (std::size_t g = 0; g < n_byte_groups; ++g) {
      const std::size_t out_offset = (block_idx * n_byte_groups + g) * kBlock;
      for (std::size_t j = 0; j < 16; ++j) {
        const std::size_t va = base_vec + kPerm0[j];
        const std::size_t vb = base_vec + kPerm0[j] + 16;
        const std::uint8_t ba = (va < n) ? codes_flat[va * n_byte_groups + g] : 0;
        const std::uint8_t bb = (vb < n) ? codes_flat[vb * n_byte_groups + g] : 0;
        blocked[out_offset + j] = static_cast<std::uint8_t>((ba >> 4) | ((bb >> 4) << 4));
        blocked[out_offset + 16 + j] =
            static_cast<std::uint8_t>((ba & 0x0F) | ((bb & 0x0F) << 4));
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
#if defined(__x86_64__) || defined(_M_X64)
  const bool kernel_exists = true;
#else
  const bool kernel_exists = (bits == 4);
#endif
  return kernel_exists && use_vector_major() && (n_byte_groups % 4 == 0);
}

std::string search_backend_name(std::size_t bits, std::size_t dim) {
  const std::size_t codes_per_byte = 8 / bits;
  const std::size_t n_byte_groups = dim / codes_per_byte;
  if (vector_major_for(bits, n_byte_groups)) {
    return "avx512_vnni";
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

void vector_major_chunk(std::span<std::uint8_t> buf) {
  std::array<std::uint8_t, kVmUnit> tmp{};
  for (std::size_t off = 0; off + kVmUnit <= buf.size(); off += kVmUnit) {
    std::memcpy(tmp.data(), buf.data() + off, kVmUnit);
    auto* unit = buf.data() + off;
    for (std::size_t j = 0; j < 4; ++j) {
      for (std::size_t v = 0; v < kBlock; ++v) {
        unit[(v / 16) * 64 + (v % 16) * 4 + j] = tmp[j * kBlock + v];
      }
    }
  }
}

void vector_major_to_seq_chunk(std::span<std::uint8_t> buf) {
  std::array<std::uint8_t, kVmUnit> tmp{};
  for (std::size_t off = 0; off + kVmUnit <= buf.size(); off += kVmUnit) {
    std::memcpy(tmp.data(), buf.data() + off, kVmUnit);
    auto* unit = buf.data() + off;
    for (std::size_t j = 0; j < 4; ++j) {
      for (std::size_t v = 0; v < kBlock; ++v) {
        unit[j * kBlock + v] = tmp[(v / 16) * 64 + (v % 16) * 4 + j];
      }
    }
  }
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

std::uint8_t read_code(std::span<const std::uint8_t> blocked, std::size_t bits,
                       std::size_t n_byte_groups, std::size_t block_idx, std::size_t g,
                       std::size_t lane) {
  const std::size_t block_base = block_idx * n_byte_groups * kBlock;
  if (vector_major_for(bits, n_byte_groups)) {
    const std::size_t idx =
        block_base + (g / 4) * 128 + (lane / 16) * 64 + (lane % 16) * 4 + (g % 4);
    return blocked[idx];
  }
#if defined(__x86_64__) || defined(_M_X64)
  return deinterleave_x86_code_byte(blocked, block_base + g * kBlock, lane);
#else
  return blocked[block_base + g * kBlock + lane];
#endif
}

}  // namespace vectorcache
