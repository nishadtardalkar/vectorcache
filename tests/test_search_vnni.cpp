#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "vectorcache/index.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/search/search_vnni.hpp"

namespace {

#if defined(_WIN32)
void set_env(const char* key, const char* value) {
  _putenv_s(key, value);
}
void unset_env(const char* key) { _putenv_s(key, ""); }
#else
void set_env(const char* key, const char* value) { setenv(key, value, 1); }
void unset_env(const char* key) { unsetenv(key); }
#endif

struct EnvGuard {
  std::string key;
  explicit EnvGuard(const char* k, const char* v) : key(k) { set_env(k, v); }
  ~EnvGuard() { unset_env(key.c_str()); }
};

std::vector<float> random_matrix(std::size_t rows, std::size_t cols, unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> out(rows * cols);
  for (float& x : out) x = dist(rng);
  return out;
}

}  // namespace

TEST(SearchVnni, SplitLutLayout) {
  constexpr std::size_t n_byte_groups = 8;
  std::vector<std::uint8_t> src(n_byte_groups * 32);
  for (std::size_t g = 0; g < n_byte_groups; ++g) {
    for (std::size_t i = 0; i < 16; ++i) {
      src[g * 32 + i] = static_cast<std::uint8_t>(0xA0 + g);       // hi
      src[g * 32 + 16 + i] = static_cast<std::uint8_t>(0x10 + g);  // lo
    }
  }
  auto split = vectorcache::split_lut_for_vnni(src, n_byte_groups);
  ASSERT_EQ(split.size(), src.size());
  // First quad: lo tables at [0..64), hi at [64..128)
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_EQ(split[j * 16], static_cast<std::uint8_t>(0x10 + j));
    EXPECT_EQ(split[64 + j * 16], static_cast<std::uint8_t>(0xA0 + j));
  }
}

TEST(SearchVnni, ParityAgainstNoVnni) {
  if (!vectorcache::use_vector_major()) {
    GTEST_SKIP() << "CPU lacks AVX-512 VNNI+VBMI; VNNI path inactive";
  }

  constexpr std::size_t dim = 64;  // 4-bit → 32 byte-groups; 2-bit → 16
  constexpr std::size_t n = 96;
  constexpr std::size_t nq = 4;
  constexpr std::size_t k = 5;

  auto db = random_matrix(n, dim, 7);
  auto queries = random_matrix(nq, dim, 11);

  vectorcache::SearchResults with_vnni;
  {
    unset_env("TURBOVEC_NO_VNNI");
    unset_env("TURBOVEC_NO_VECTOR_MAJOR");
    ASSERT_TRUE(vectorcache::use_vector_major());
    vectorcache::TurboQuantIndex index(dim, 4);
    index.add(db);
    index.prepare();
    with_vnni = index.search(queries, k);
  }

  vectorcache::SearchResults without_vnni;
  {
    EnvGuard guard("TURBOVEC_NO_VNNI", "1");
    ASSERT_FALSE(vectorcache::use_vector_major());
    vectorcache::TurboQuantIndex index(dim, 4);
    index.add(db);
    index.prepare();
    without_vnni = index.search(queries, k);
  }

  ASSERT_EQ(with_vnni.nq, without_vnni.nq);
  ASSERT_EQ(with_vnni.k, without_vnni.k);
  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      const std::size_t off = qi * k + j;
      EXPECT_EQ(with_vnni.ids[off], without_vnni.ids[off]) << "qi=" << qi << " j=" << j;
      EXPECT_NEAR(with_vnni.scores[off], without_vnni.scores[off], 1e-3f)
          << "qi=" << qi << " j=" << j;
    }
  }
}

TEST(SearchVnni, ParityBits2) {
  if (!vectorcache::use_vector_major()) {
    GTEST_SKIP() << "CPU lacks AVX-512 VNNI+VBMI; VNNI path inactive";
  }

  constexpr std::size_t dim = 64;  // 2-bit → 16 byte-groups (%4==0)
  constexpr std::size_t n = 64;
  constexpr std::size_t nq = 2;
  constexpr std::size_t k = 3;

  auto db = random_matrix(n, dim, 19);
  auto queries = random_matrix(nq, dim, 23);

  vectorcache::SearchResults with_vnni;
  {
    unset_env("TURBOVEC_NO_VNNI");
    vectorcache::TurboQuantIndex index(dim, 2);
    index.add(db);
    index.prepare();
    with_vnni = index.search(queries, k);
  }

  vectorcache::SearchResults without_vnni;
  {
    EnvGuard guard("TURBOVEC_NO_VNNI", "1");
    vectorcache::TurboQuantIndex index(dim, 2);
    index.add(db);
    index.prepare();
    without_vnni = index.search(queries, k);
  }

  for (std::size_t qi = 0; qi < nq; ++qi) {
    for (std::size_t j = 0; j < k; ++j) {
      const std::size_t off = qi * k + j;
      EXPECT_EQ(with_vnni.ids[off], without_vnni.ids[off]);
      EXPECT_NEAR(with_vnni.scores[off], without_vnni.scores[off], 1e-3f);
    }
  }
}
