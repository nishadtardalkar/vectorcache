#include "vectorcache/ingest/timing.hpp"

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

}  // namespace vectorcache::ingest
