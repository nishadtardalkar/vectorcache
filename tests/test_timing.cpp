#include <gtest/gtest.h>

#include "vectorcache/ingest/timing.hpp"

using namespace vectorcache::ingest;

TEST(TimingTest, SummaryStats) {
  IngestTimings timings;
  timings.record(100);
  timings.record(200);
  timings.record(300);
  const auto summary = timings.summary();
  EXPECT_EQ(summary.total_ns, 600u);
  EXPECT_DOUBLE_EQ(summary.mean_ns, 200.0);
  EXPECT_EQ(summary.min_ns, 100u);
  EXPECT_EQ(summary.max_ns, 300u);
  EXPECT_EQ(summary.vectors, 3u);
}

TEST(TimingTest, EmptySummary) {
  IngestTimings timings;
  const auto summary = timings.summary();
  EXPECT_EQ(summary.vectors, 0u);
  EXPECT_EQ(summary.total_ns, 0u);
}
