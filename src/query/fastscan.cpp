#include "vectorcache/query/fastscan.hpp"

#include <cstring>

#include "vectorcache/error.hpp"
#include "vectorcache/quantize/quantize.hpp"

namespace vectorcache::query {

void BlockedCodes::clear() {
  n_ = 0;
  bits_ = 0;
  num_groups_ = 0;
  words_per_vec_ = 0;
  n_blocks_ = 0;
  bytes_.clear();
  bit1_words_.clear();
}

void BlockedCodes::rebuild(std::span<const std::uint64_t> codes, std::size_t words_per_vec,
                           std::size_t n, std::size_t dim, std::size_t bits,
                           std::size_t block_dims) {
  if (n == 0) {
    clear();
    return;
  }
  if (bits != 1 && bits != 2 && bits != 4 && bits != 8) {
    clear();
    return;
  }
  if (dim % block_dims != 0) {
    clear();
    return;
  }
  const std::size_t m = dim / block_dims;
  if ((m * bits) % 8 != 0) {
    clear();
    return;
  }
  if (words_per_vec != quantize::l0_words_per_vector(dim, bits, block_dims)) {
    throw Error("BlockedCodes::rebuild: words_per_vec mismatch");
  }
  if (codes.size() < n * words_per_vec) {
    throw Error("BlockedCodes::rebuild: codes too small");
  }

  n_ = n;
  bits_ = bits;
  words_per_vec_ = words_per_vec;
  num_groups_ = (m * bits) / 8;
  n_blocks_ = (n + kBlock - 1) / kBlock;

  bytes_.assign(n_blocks_ * num_groups_ * kBlock, 0);
  const auto* src_bytes = reinterpret_cast<const std::uint8_t*>(codes.data());
  const std::size_t bytes_per_vec = words_per_vec * sizeof(std::uint64_t);

  for (std::size_t v = 0; v < n; ++v) {
    const std::size_t block = v / kBlock;
    const std::size_t vin = v % kBlock;
    const std::uint8_t* row = src_bytes + v * bytes_per_vec;
    for (std::size_t g = 0; g < num_groups_; ++g) {
      bytes_[(block * num_groups_ + g) * kBlock + vin] = row[g];
    }
  }

  bit1_words_.clear();
  if (bits == 1 && (m % 64) == 0) {
    bit1_words_.assign(n_blocks_ * words_per_vec_ * kBlock, 0);
    for (std::size_t v = 0; v < n; ++v) {
      const std::size_t block = v / kBlock;
      const std::size_t vin = v % kBlock;
      const std::uint64_t* row = codes.data() + v * words_per_vec;
      for (std::size_t w = 0; w < words_per_vec; ++w) {
        bit1_words_[(block * words_per_vec + w) * kBlock + vin] = row[w];
      }
    }
  }
}

const std::uint8_t* BlockedCodes::group_bytes(std::size_t block, std::size_t group) const {
  if (block >= n_blocks_ || group >= num_groups_) {
    throw Error("BlockedCodes::group_bytes out of range");
  }
  return bytes_.data() + (block * num_groups_ + group) * kBlock;
}

const std::uint64_t* BlockedCodes::bit1_word_column(std::size_t block, std::size_t word) const {
  if (!has_bit1_words() || block >= n_blocks_ || word >= words_per_vec_) {
    throw Error("BlockedCodes::bit1_word_column out of range");
  }
  return bit1_words_.data() + (block * words_per_vec_ + word) * kBlock;
}

std::size_t BlockedCodes::block_count(std::size_t block) const {
  if (block >= n_blocks_) {
    return 0;
  }
  if (block + 1 < n_blocks_) {
    return kBlock;
  }
  const std::size_t rem = n_ % kBlock;
  return rem == 0 ? kBlock : rem;
}

void BlockedCodes::unpack_vector_bytes(std::size_t index, std::span<std::uint8_t> out) const {
  if (index >= n_ || out.size() < num_groups_) {
    throw Error("BlockedCodes::unpack_vector_bytes invalid args");
  }
  const std::size_t block = index / kBlock;
  const std::size_t vin = index % kBlock;
  for (std::size_t g = 0; g < num_groups_; ++g) {
    out[g] = bytes_[(block * num_groups_ + g) * kBlock + vin];
  }
}

}  // namespace vectorcache::query
