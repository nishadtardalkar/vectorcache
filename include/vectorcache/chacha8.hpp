#pragma once

#include <array>
#include <cstdint>

namespace vectorcache {

/// ChaCha8 stream matching `rand_chacha = "=0.3.1"` ChaCha8Rng.
class ChaCha8Rng {
 public:
  explicit ChaCha8Rng(const std::array<std::uint8_t, 32>& seed);

  std::uint32_t next_u32();
  std::uint64_t next_u64();

 private:
  void refill();

  std::array<std::uint32_t, 16> state_{};
  std::array<std::uint32_t, 16> buffer_{};
  int index_ = 16;
};

}  // namespace vectorcache
