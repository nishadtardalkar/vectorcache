#pragma once

#include <cstddef>

#ifndef VECTORCACHE_QUERY_DEPTH
#define VECTORCACHE_QUERY_DEPTH 3
#endif

#if VECTORCACHE_QUERY_DEPTH < 1 || VECTORCACHE_QUERY_DEPTH > 3
#error VECTORCACHE_QUERY_DEPTH must be 1, 2, or 3
#endif

namespace vectorcache::query {

inline constexpr std::size_t query_depth() {
  return static_cast<std::size_t>(VECTORCACHE_QUERY_DEPTH);
}

}  // namespace vectorcache::query
