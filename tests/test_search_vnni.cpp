#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "vectorcache/index.hpp"
#include "vectorcache/pack/pack.hpp"
#include "vectorcache/quantize/codebook.hpp"
#include "vectorcache/search/search_vnni.hpp"

namespace {

#if defined(_WIN32)
void set_env(const char* key, const char* value) { _putenv_s(key, value); }
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

void expect_same_results(const vectorcache::SearchResults& a, const vectorcache::SearchResults& b,
                         float score_tol = 1e-4f) {
  ASSERT_EQ(a.nq, b.nq);
  ASSERT_EQ(a.k, b.k);
  for (std::size_t i = 0; i < a.nq * a.k; ++i) {
    EXPECT_EQ(a.ids[i], b.ids[i]) << "i=" << i;
    EXPECT_NEAR(a.scores[i], b.scores[i], score_tol) << "i=" << i;
  }
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
  for (std::size_t j = 0; j < 4; ++j) {
    EXPECT_EQ(split[j * 16], static_cast<std::uint8_t>(0x10 + j));
    EXPECT_EQ(split[64 + j * 16], static_cast<std::uint8_t>(0xA0 + j));
  }
}

TEST(SearchVnni, BuildPermuteDotWeightSlots) {
  constexpr std::size_t dim = 16;  // 8 byte-groups
  std::vector<float> q(dim);
  for (std::size_t d = 0; d < dim; ++d) q[d] = static_cast<float>(d + 1);
  auto cb = vectorcache::codebook(4, dim);
  auto pd = vectorcache::build_permute_dot(q, cb.second, dim);
  ASSERT_EQ(pd.weights.size(), (dim / 2) * 2u);
  // Group g=0: lo = odd dim 1, hi = even dim 0; slots q4=0, j=0.
  // Weights are int8 quantized; check relative packing order via rebuild at large scale.
  // Direct slot map: weights[q4*8+j]=lo(2g+1), weights[q4*8+4+j]=hi(2g).
  for (std::size_t g = 0; g < dim / 2; ++g) {
    const std::size_t q4 = g / 4;
    const std::size_t j = g % 4;
    // Signs should match query signs after quantization with positive qs.
    EXPECT_GE(pd.weights[q4 * 8 + j], 0) << "lo g=" << g;
    EXPECT_GE(pd.weights[q4 * 8 + 4 + j], 0) << "hi g=" << g;
  }
  // Increasing query magnitudes → non-decreasing |weights| within a nibble group of 4.
  EXPECT_LE(std::abs(static_cast<int>(pd.weights[0])), std::abs(static_cast<int>(pd.weights[1])));
}

TEST(SearchVnni, Bits4PermuteDotSmokeAndBatchIdentity) {
  if (!vectorcache::use_vector_major()) {
    GTEST_SKIP() << "CPU lacks AVX-512 VNNI+VBMI; permute-dot path inactive";
  }

  unset_env("TURBOVEC_NO_VNNI");
  unset_env("TURBOVEC_NO_VECTOR_MAJOR");
#if defined(_WIN32) && defined(__GNUC__) && !defined(__clang__)
  EXPECT_EQ(vectorcache::search_backend_name(4, 64), "avx512_vnni");
#else
  EXPECT_EQ(vectorcache::search_backend_name(4, 64), "avx512_permute_dot");
#endif

  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 96;
  constexpr std::size_t nq = 8;
  constexpr std::size_t k = 5;

  auto db = random_matrix(n, dim, 7);
  auto queries = random_matrix(nq, dim, 11);

  vectorcache::SearchResults multi;
  {
    vectorcache::TurboQuantIndex index(dim, 4);
    index.add(db);
    index.prepare();
    multi = index.search(queries, k);
  }

  // Serial: one query at a time (same kernel family, different tiling).
  vectorcache::SearchResults serial;
  serial.nq = nq;
  serial.k = k;
  serial.scores.resize(nq * k);
  serial.ids.resize(nq * k);
  {
    vectorcache::TurboQuantIndex index(dim, 4);
    index.add(db);
    index.prepare();
    for (std::size_t qi = 0; qi < nq; ++qi) {
      auto one = index.search(std::span<const float>(queries.data() + qi * dim, dim), k);
      for (std::size_t j = 0; j < k; ++j) {
        serial.scores[qi * k + j] = one.scores[j];
        serial.ids[qi * k + j] = one.ids[j];
      }
    }
  }

  for (std::size_t i = 0; i < nq * k; ++i) {
    EXPECT_TRUE(std::isfinite(multi.scores[i]));
  }
  expect_same_results(multi, serial, 1e-3f);
}

TEST(SearchVnni, ParityBits2) {
  if (!vectorcache::use_vector_major()) {
    GTEST_SKIP() << "CPU lacks AVX-512 VNNI+VBMI; VNNI path inactive";
  }

  EXPECT_EQ(vectorcache::search_backend_name(2, 64), "avx512_vnni");

  constexpr std::size_t dim = 64;
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
    // use_vector_major may already be cached true; force layout via NO_VECTOR_MAJOR too.
    EnvGuard guard2("TURBOVEC_NO_VECTOR_MAJOR", "1");
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

TEST(SearchVnni, Bits2BatchMatchesSerial) {
  if (!vectorcache::use_vector_major()) {
    GTEST_SKIP() << "CPU lacks AVX-512 VNNI+VBMI; VNNI path inactive";
  }

  unset_env("TURBOVEC_NO_VNNI");
  unset_env("TURBOVEC_NO_VECTOR_MAJOR");

  constexpr std::size_t dim = 64;
  constexpr std::size_t n = 128;
  constexpr std::size_t nq = 8;
  constexpr std::size_t k = 4;

  auto db = random_matrix(n, dim, 31);
  auto queries = random_matrix(nq, dim, 37);

  vectorcache::SearchResults multi;
  {
    vectorcache::TurboQuantIndex index(dim, 2);
    index.add(db);
    index.prepare();
    multi = index.search(queries, k);
  }

  vectorcache::SearchResults serial;
  serial.nq = nq;
  serial.k = k;
  serial.scores.resize(nq * k);
  serial.ids.resize(nq * k);
  {
    vectorcache::TurboQuantIndex index(dim, 2);
    index.add(db);
    index.prepare();
    for (std::size_t qi = 0; qi < nq; ++qi) {
      auto one = index.search(std::span<const float>(queries.data() + qi * dim, dim), k);
      for (std::size_t j = 0; j < k; ++j) {
        serial.scores[qi * k + j] = one.scores[j];
        serial.ids[qi * k + j] = one.ids[j];
      }
    }
  }

  expect_same_results(multi, serial, 1e-3f);
}
