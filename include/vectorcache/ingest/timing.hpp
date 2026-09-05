#pragma once

#include <cstdint>
#include <string>

namespace vectorcache::ingest {

struct TimingSummary {
  static std::string format_duration(std::uint64_t ns);
};

}  // namespace vectorcache::ingest
