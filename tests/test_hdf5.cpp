#include <filesystem>
#include <gtest/gtest.h>

#include "vectorcache/datasets/hdf5.hpp"

TEST(Hdf5Test, GloveIntegrationIfPresent) {
  const std::filesystem::path path = "data/glove-200-angular.hdf5";
  if (!std::filesystem::is_regular_file(path)) {
    GTEST_SKIP() << "GloVe HDF5 not present";
  }

  auto reader = vectorcache::datasets::Hdf5GloveReader::open(
      path, vectorcache::datasets::DatasetSplit::Train);
  const auto meta = reader.meta();
  EXPECT_EQ(meta.dim, 200u);
  EXPECT_GT(meta.count, 0u);

  std::vector<float> v(200);
  EXPECT_TRUE(reader.next_vector_into(v));
  EXPECT_EQ(v.size(), 200u);
}
