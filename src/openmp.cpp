#include "vectorcache/openmp.hpp"

#include <cstdlib>
#include <thread>

#include "vectorcache/error.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace vectorcache {

void configure_openmp_threads(std::optional<int> explicit_threads) {
#ifdef _OPENMP
  if (explicit_threads.has_value()) {
    if (*explicit_threads < 1) {
      throw Error("--threads must be at least 1");
    }
    omp_set_num_threads(*explicit_threads);
    return;
  }
  if (std::getenv("OMP_NUM_THREADS") != nullptr) {
    return;
  }
  const int n = static_cast<int>(std::thread::hardware_concurrency());
  if (n > 0) {
    omp_set_num_threads(n);
  }
#else
  (void)explicit_threads;
#endif
}

int openmp_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

}  // namespace vectorcache
