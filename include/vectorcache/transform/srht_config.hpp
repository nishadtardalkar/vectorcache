#pragma once

#include <cstddef>

#ifndef VECTORCACHE_SRHT_ROUNDS
#define VECTORCACHE_SRHT_ROUNDS 1
#endif

#if VECTORCACHE_SRHT_ROUNDS < 1 || VECTORCACHE_SRHT_ROUNDS > 3
#error VECTORCACHE_SRHT_ROUNDS must be 1, 2, or 3
#endif

namespace vectorcache::transform {

inline constexpr std::size_t srht_rounds() {
  return static_cast<std::size_t>(VECTORCACHE_SRHT_ROUNDS);
}

}  // namespace vectorcache::transform
