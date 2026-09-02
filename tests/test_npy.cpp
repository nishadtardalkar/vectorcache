#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include "vectorcache/datasets/npy.hpp"

TEST(NpyTest, RoundTripSmallNpy) {
  const auto dir = std::filesystem::temp_directory_path() / "vectorcache_npy_test";
  std::filesystem::create_directories(dir);
  const auto path = dir / "test.npy";

  const std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  vectorcache::datasets::write_npy_f32_matrix(path, 3, 2, data.data());

  {
    auto reader = vectorcache::datasets::NpyReader::open(path, "test");
    EXPECT_EQ(reader.meta().count, 3u);
    EXPECT_EQ(reader.meta().dim, 2u);

    std::vector<float> buf(2);
    EXPECT_TRUE(reader.next_vector_into(buf));
    EXPECT_FLOAT_EQ(buf[0], 1.0f);
    EXPECT_FLOAT_EQ(buf[1], 2.0f);
    EXPECT_TRUE(reader.next_vector_into(buf));
    EXPECT_FLOAT_EQ(buf[0], 3.0f);
    EXPECT_FLOAT_EQ(buf[1], 4.0f);
    EXPECT_TRUE(reader.next_vector_into(buf));
    EXPECT_FLOAT_EQ(buf[0], 5.0f);
    EXPECT_FLOAT_EQ(buf[1], 6.0f);
    EXPECT_FALSE(reader.next_vector_into(buf));
  }

  std::filesystem::remove_all(dir);
}
