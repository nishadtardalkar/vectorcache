#pragma once

#ifndef __AVX512F__
#error "vectorcache requires AVX-512F (compile with -mavx512f or /arch:AVX512)"
#endif

#include <cstddef>
#include <immintrin.h>

namespace vectorcache::simd {

inline constexpr std::size_t kWidth = 16;

inline __mmask16 tail_mask(std::size_t remaining) {
  return remaining >= kWidth ? static_cast<__mmask16>(0xFFFF)
                             : static_cast<__mmask16>((1u << remaining) - 1u);
}

}  // namespace vectorcache::simd
