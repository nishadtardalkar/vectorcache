#pragma once

#include <cstddef>

namespace vectorcache {

/// FastScan block width: 192 vectors × 1 byte-group tiles (not turbovec 32×4).
inline constexpr std::size_t kBlock = 192;
/// Sign-plane / true-VNNI block width (turbovec BLOCK): 32 vectors × 4-group VM tiles.
inline constexpr std::size_t kSignBlock = 32;
/// Bytes per vector-major unit: 4 byte-groups × 32 vectors.
inline constexpr std::size_t kSignVmUnit = 4 * kSignBlock;
/// SIMD accumulate/flush chunk inside a block (2×16 AVX-512 lanes).
inline constexpr std::size_t kFlushChunk = 32;
inline constexpr std::size_t kChunksPerBlock = kBlock / kFlushChunk;  // 6
inline constexpr std::size_t kMaxDim = 16384;
/// One FWHT round (perm + signs + block WH) before 4-bit quant — SRHT stand-in.
inline constexpr std::size_t kRotationRounds = 1;
inline constexpr float kMinInputNorm = 1e-10f;
inline constexpr float kMaxInputMagnitude = 1e16f;
inline constexpr double kDegenerateInnerEps = 1e-12;
inline constexpr std::size_t kNormChains = 8;
inline constexpr std::size_t kFlushEvery = 256;
inline constexpr std::size_t kMinCalibrationRows = 2;
inline constexpr std::size_t kRecommendedCalibrationRows = 1000;
inline constexpr std::size_t kDefaultLandmarks = 1024;

}  // namespace vectorcache
