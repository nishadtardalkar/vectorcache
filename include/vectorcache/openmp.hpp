#pragma once

#include <optional>

namespace vectorcache {

/// Configure OpenMP worker count before parallel ingestion.
/// Precedence: explicit_threads > OMP_NUM_THREADS env > hardware_concurrency().
void configure_openmp_threads(std::optional<int> explicit_threads = std::nullopt);

/// Max threads for the next parallel region (1 when OpenMP is disabled).
int openmp_max_threads();

}  // namespace vectorcache
