#include <gtest/gtest.h>

#include "vectorcache/datasets/sample.hpp"

using namespace vectorcache::datasets;

namespace {

class SeqReader : public DatasetReader {
 public:
  explicit SeqReader(std::size_t count, std::size_t dim) : dim_(dim), count_(count) {}

  DatasetMeta meta() const override { return {dim_, count_, "seq"}; }

  bool next_vector_into(std::span<float> out) override {
    if (index_ >= count_) {
      return false;
    }
    for (std::size_t i = 0; i < dim_; ++i) {
      out[i] = static_cast<float>(index_);
    }
    ++index_;
    return true;
  }

 private:
  std::size_t dim_;
  std::size_t count_;
  std::size_t index_ = 0;
};

}  // namespace

TEST(SampleTest, IndexSubsetReadsRequestedRows) {
  SeqReader seq(10, 2);
  const auto indices = sample_index_range(2, 10, 3, 42);
  IndexSubsetReader subset(seq, indices);

  std::vector<float> buf(2);
  std::vector<std::size_t> seen;
  while (subset.next_vector_into(buf)) {
    seen.push_back(static_cast<std::size_t>(buf[0]));
  }

  ASSERT_EQ(seen.size(), indices.size());
  for (std::size_t i = 0; i < indices.size(); ++i) {
    EXPECT_EQ(seen[i], indices[i]);
  }
}

TEST(SampleTest, LimitedReaderCapsCount) {
  SeqReader seq(100, 1);
  LimitedReader limited(seq, 5);
  std::vector<float> buf(1);
  std::size_t n = 0;
  while (limited.next_vector_into(buf)) {
    ++n;
  }
  EXPECT_EQ(n, 5u);
}

TEST(SampleTest, SampleRangeIsSorted) {
  const auto idx = sample_index_range(100, 1000, 50, 7);
  EXPECT_EQ(idx.size(), 50u);
  EXPECT_TRUE(std::is_sorted(idx.begin(), idx.end()));
  EXPECT_GE(idx.front(), 100u);
  EXPECT_LT(idx.back(), 1000u);
}
