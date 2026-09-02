#pragma once

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <new>
#include <vector>

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace vectorcache {

template <typename T, std::size_t Align>
struct AlignedAllocator {
  using value_type = T;

  AlignedAllocator() noexcept = default;

  template <typename U>
  AlignedAllocator(const AlignedAllocator<U, Align>&) noexcept {}

  template <typename U>
  struct rebind {
    using other = AlignedAllocator<U, Align>;
  };

  [[nodiscard]] T* allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      throw std::bad_alloc();
    }
    if (n == 0) {
      return nullptr;
    }
    void* ptr = nullptr;
#if defined(_WIN32)
    ptr = _aligned_malloc(n * sizeof(T), Align);
#else
    if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0) {
      ptr = nullptr;
    }
#endif
    if (ptr == nullptr) {
      throw std::bad_alloc();
    }
    return static_cast<T*>(ptr);
  }

  void deallocate(T* ptr, std::size_t) noexcept {
    if (ptr == nullptr) {
      return;
    }
#if defined(_WIN32)
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
  }
};

template <typename T, typename U, std::size_t Align>
bool operator==(const AlignedAllocator<T, Align>&, const AlignedAllocator<U, Align>&) noexcept {
  return true;
}

template <typename T, typename U, std::size_t Align>
bool operator!=(const AlignedAllocator<T, Align>& lhs, const AlignedAllocator<U, Align>& rhs) noexcept {
  return !(lhs == rhs);
}

template <typename T, std::size_t Align = 64>
using AlignedVector = std::vector<T, AlignedAllocator<T, Align>>;

}  // namespace vectorcache
