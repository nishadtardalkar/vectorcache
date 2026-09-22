#include "vectorcache/chacha8.hpp"

#include <cstring>

namespace vectorcache {
namespace {

constexpr std::uint32_t rotl32(std::uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

void quarter_round(std::uint32_t& a, std::uint32_t& b, std::uint32_t& c, std::uint32_t& d) {
  a += b;
  d ^= a;
  d = rotl32(d, 16);
  c += d;
  b ^= c;
  b = rotl32(b, 12);
  a += b;
  d ^= a;
  d = rotl32(d, 8);
  c += d;
  b ^= c;
  b = rotl32(b, 7);
}

std::uint32_t load_le32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

}  // namespace

ChaCha8Rng::ChaCha8Rng(const std::array<std::uint8_t, 32>& seed) {
  // IETF ChaCha20 layout used by rand_chacha 0.3.x.
  state_[0] = 0x61707865u;
  state_[1] = 0x3320646eu;
  state_[2] = 0x79622d32u;
  state_[3] = 0x6b206574u;
  for (int i = 0; i < 8; ++i) {
    state_[4 + i] = load_le32(seed.data() + static_cast<std::size_t>(i) * 4);
  }
  state_[12] = 0;  // counter lo
  state_[13] = 0;  // counter hi
  state_[14] = 0;  // nonce
  state_[15] = 0;
  index_ = 16;
}

void ChaCha8Rng::refill() {
  std::array<std::uint32_t, 16> working = state_;
  // ChaCha8 = 8 rounds = 4 double-rounds.
  for (int i = 0; i < 4; ++i) {
    quarter_round(working[0], working[4], working[8], working[12]);
    quarter_round(working[1], working[5], working[9], working[13]);
    quarter_round(working[2], working[6], working[10], working[14]);
    quarter_round(working[3], working[7], working[11], working[15]);
    quarter_round(working[0], working[5], working[10], working[15]);
    quarter_round(working[1], working[6], working[11], working[12]);
    quarter_round(working[2], working[7], working[8], working[13]);
    quarter_round(working[3], working[4], working[9], working[14]);
  }
  for (int i = 0; i < 16; ++i) {
    buffer_[static_cast<std::size_t>(i)] = working[static_cast<std::size_t>(i)] +
                                           state_[static_cast<std::size_t>(i)];
  }
  // Increment 64-bit counter in state_[12], state_[13].
  if (++state_[12] == 0) {
    ++state_[13];
  }
  index_ = 0;
}

std::uint32_t ChaCha8Rng::next_u32() {
  if (index_ >= 16) {
    refill();
  }
  return buffer_[static_cast<std::size_t>(index_++)];
}

std::uint64_t ChaCha8Rng::next_u64() {
  const std::uint64_t lo = next_u32();
  const std::uint64_t hi = next_u32();
  return lo | (hi << 32);
}

}  // namespace vectorcache
