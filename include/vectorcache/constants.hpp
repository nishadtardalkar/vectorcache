#pragma once

#include <cstddef>

namespace vectorcache {

inline constexpr std::size_t kBlock = 32;
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
