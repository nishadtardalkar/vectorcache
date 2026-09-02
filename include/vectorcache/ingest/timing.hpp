#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vectorcache::ingest {

struct TimingSummary {
  std::uint64_t total_ns = 0;
  double mean_ns = 0.0;
  std::uint64_t min_ns = 0;
  std::uint64_t max_ns = 0;
  std::uint64_t vectors = 0;

  static std::string format_duration(std::uint64_t ns);
};

class IngestTimings {
 public:
  void record(std::uint64_t elapsed_ns);
  TimingSummary summary() const;

 private:
  std::vector<std::uint64_t> per_vector_ns_;
};

}  // namespace vectorcache::ingest
