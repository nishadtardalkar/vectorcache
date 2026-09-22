#pragma once

#include <cstddef>

namespace vectorcache {

inline constexpr std::size_t kBlock = 32;
inline constexpr std::size_t kMaxDim = 16384;
inline constexpr std::size_t kRotationRounds = 2;
inline constexpr float kMinInputNorm = 1e-10f;
inline constexpr float kMaxInputMagnitude = 1e16f;
inline constexpr double kDegenerateInnerEps = 0.1;
inline constexpr std::size_t kMinCalibrationRows = 2;
inline constexpr std::size_t kRecommendedCalibrationRows = 1000;
inline constexpr std::size_t kNormChains = 8;
inline constexpr std::size_t kFlushEvery = 256;

}  // namespace vectorcache
