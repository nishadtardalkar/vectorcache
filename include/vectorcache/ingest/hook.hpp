#pragma once

#include <cstdint>
#include <span>

namespace vectorcache::ingest {

/// Per-vector callback invoked during ingestion.
class VectorHook {
 public:
  virtual ~VectorHook() = default;
  virtual void on_vector(std::uint64_t global_id, std::span<const float> vector) = 0;
};

/// Default hook that performs no processing.
class NoopHook : public VectorHook {
 public:
  void on_vector(std::uint64_t /*global_id*/, std::span<const float> /*vector*/) override {}
};

}  // namespace vectorcache::ingest
