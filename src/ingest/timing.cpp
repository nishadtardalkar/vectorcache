#include "vectorcache/ingest/timing.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace vectorcache::ingest {

std::string TimingSummary::format_duration(std::uint64_t ns) {
  std::ostringstream oss;
  if (ns >= 1'000'000'000ULL) {
    oss.precision(3);
    oss << std::fixed << (static_cast<double>(ns) / 1'000'000'000.0) << "s";
  } else if (ns >= 1'000'000ULL) {
    oss.precision(3);
    oss << std::fixed << (static_cast<double>(ns) / 1'000'000.0) << "ms";
  } else if (ns >= 1'000ULL) {
    oss.precision(3);
    oss << std::fixed << (static_cast<double>(ns) / 1'000.0) << "us";
  } else {
    oss << ns << "ns";
  }
  return oss.str();
}

void IngestTimings::record(std::uint64_t elapsed_ns) {
  per_vector_ns_.push_back(elapsed_ns);
}

TimingSummary IngestTimings::summary() const {
  TimingSummary result;
  if (per_vector_ns_.empty()) {
    return result;
  }

  std::uint64_t total_ns = 0;
  std::uint64_t min_ns = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t max_ns = 0;
  for (const std::uint64_t ns : per_vector_ns_) {
    total_ns += ns;
    min_ns = std::min(min_ns, ns);
    max_ns = std::max(max_ns, ns);
  }

  result.total_ns = total_ns;
  result.vectors = per_vector_ns_.size();
  result.mean_ns = static_cast<double>(total_ns) / static_cast<double>(result.vectors);
  result.min_ns = min_ns;
  result.max_ns = max_ns;
  return result;
}

}  // namespace vectorcache::ingest
