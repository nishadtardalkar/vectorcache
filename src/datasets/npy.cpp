#include "vectorcache/datasets/npy.hpp"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <sstream>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#include "vectorcache/error.hpp"

namespace vectorcache::datasets {

namespace {

struct ParsedHeader {
  std::size_t count;
  std::size_t dim;
  std::size_t data_offset;
};

ParsedHeader parse_npy_f32_matrix(const char* data, std::size_t size) {
  static constexpr char kMagic[] = "\x93NUMPY";
  if (size < 10) {
    throw Error("NPY file too small");
  }
  if (std::memcmp(data, kMagic, 6) != 0) {
    throw Error("invalid NPY magic bytes");
  }

  const unsigned char major = static_cast<unsigned char>(data[6]);
  const unsigned char minor = static_cast<unsigned char>(data[7]);
  std::size_t header_len = 0;
  std::size_t header_start = 0;

  if (major == 1 && minor == 0) {
    if (size < 10) {
      throw Error("NPY v1 header truncated");
    }
    header_len = static_cast<std::size_t>(static_cast<unsigned char>(data[8])) |
                 (static_cast<std::size_t>(static_cast<unsigned char>(data[9])) << 8);
    header_start = 10;
  } else if (major == 2 && minor == 0) {
    if (size < 12) {
      throw Error("NPY v2 header truncated");
    }
    header_len = static_cast<std::size_t>(static_cast<unsigned char>(data[8])) |
                 (static_cast<std::size_t>(static_cast<unsigned char>(data[9])) << 8) |
                 (static_cast<std::size_t>(static_cast<unsigned char>(data[10])) << 16) |
                 (static_cast<std::size_t>(static_cast<unsigned char>(data[11])) << 24);
    header_start = 12;
  } else {
    throw Error("unsupported NPY version " + std::to_string(major) + "." + std::to_string(minor));
  }

  const std::size_t header_end = header_start + header_len;
  if (header_end > size) {
    throw Error("NPY header extends past end of file");
  }

  const std::string header(data + header_start, header_len);
  if (header.find("'descr': '<f4'") == std::string::npos &&
      header.find("\"descr\": \"<f4\"") == std::string::npos) {
    throw Error("expected float32 ('<f4') NPY array");
  }

  const auto shape_pos = header.find("shape");
  if (shape_pos == std::string::npos) {
    throw Error("missing 'shape' in NPY header");
  }
  const auto open_paren = header.find('(', shape_pos);
  const auto close_paren = header.find(')', open_paren);
  if (open_paren == std::string::npos || close_paren == std::string::npos) {
    throw Error("failed to parse NPY shape");
  }

  std::vector<std::size_t> shape;
  std::istringstream shape_stream(header.substr(open_paren + 1, close_paren - open_paren - 1));
  std::string token;
  while (std::getline(shape_stream, token, ',')) {
    token.erase(0, token.find_first_not_of(" \t"));
    token.erase(token.find_last_not_of(" \t") + 1);
    if (token.empty()) {
      continue;
    }
    shape.push_back(static_cast<std::size_t>(std::stoull(token)));
  }
  if (shape.size() != 2) {
    throw Error("expected 2-D NPY matrix");
  }

  const std::size_t count = shape[0];
  const std::size_t dim = shape[1];
  const std::size_t expected_bytes = header_end + count * dim * 4;
  if (size < expected_bytes) {
    throw Error("NPY data truncated");
  }

  return {count, dim, header_end};
}

#if defined(_WIN32)

std::vector<char> read_file_owned(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw Error("failed to open " + path.string());
  }
  file.seekg(0, std::ios::end);
  const std::size_t file_size = static_cast<std::size_t>(file.tellg());
  file.seekg(0, std::ios::beg);
  std::vector<char> data(file_size);
  file.read(data.data(), static_cast<std::streamsize>(file_size));
  return data;
}

#endif

}  // namespace

namespace detail {

#if !defined(_WIN32)

MmapRegion::MmapRegion(MmapRegion&& other) noexcept : data(other.data), size(other.size) {
  other.data = nullptr;
  other.size = 0;
}

MmapRegion& MmapRegion::operator=(MmapRegion&& other) noexcept {
  if (this != &other) {
    if (data != nullptr) {
      ::munmap(data, size);
    }
    data = other.data;
    size = other.size;
    other.data = nullptr;
    other.size = 0;
  }
  return *this;
}

MmapRegion::~MmapRegion() {
  if (data != nullptr) {
    ::munmap(data, size);
  }
}

MmapRegion mmap_file(const std::filesystem::path& path) {
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) {
    throw Error("failed to open " + path.string());
  }
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
    ::close(fd);
    throw Error("failed to stat " + path.string());
  }
  const std::size_t file_size = static_cast<std::size_t>(st.st_size);
  void* mapped = ::mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  ::close(fd);
  if (mapped == MAP_FAILED) {
    throw Error("failed to mmap " + path.string());
  }
  MmapRegion region;
  region.data = mapped;
  region.size = file_size;
  return region;
}

#else

MmapRegion::MmapRegion(MmapRegion&& other) noexcept : data(other.data), size(other.size) {
  other.data = nullptr;
  other.size = 0;
}

MmapRegion& MmapRegion::operator=(MmapRegion&& other) noexcept {
  if (this != &other) {
    if (data != nullptr) {
      ::UnmapViewOfFile(data);
    }
    data = other.data;
    size = other.size;
    other.data = nullptr;
    other.size = 0;
  }
  return *this;
}

MmapRegion::~MmapRegion() {
  if (data != nullptr) {
    ::UnmapViewOfFile(data);
  }
}

MmapRegion mmap_file(const std::filesystem::path& path) {
  HANDLE file = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) {
    throw Error("failed to open " + path.string());
  }

  LARGE_INTEGER file_size_li {};
  if (!::GetFileSizeEx(file, &file_size_li)) {
    ::CloseHandle(file);
    throw Error("failed to stat " + path.string());
  }
  const std::size_t file_size = static_cast<std::size_t>(file_size_li.QuadPart);

  HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  ::CloseHandle(file);
  if (mapping == nullptr) {
    throw Error("failed to mmap " + path.string());
  }

  void* view = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
  ::CloseHandle(mapping);
  if (view == nullptr) {
    throw Error("failed to mmap " + path.string());
  }

  MmapRegion region;
  region.data = view;
  region.size = file_size;
  return region;
}

#endif

}  // namespace detail

NpyReader::NpyReader(std::vector<char> owned_data, detail::MmapRegion mmap, const char* label,
                     std::size_t dim, std::size_t count, std::size_t data_offset)
    : owned_data_(std::move(owned_data)),
      mmap_(std::move(mmap)),
      label_(label),
      dim_(dim),
      count_(count),
      index_(0),
      data_offset_(data_offset) {}

const char* NpyReader::data_ptr() const {
  if (!owned_data_.empty()) {
    return owned_data_.data();
  }
  return static_cast<const char*>(mmap_.data);
}

std::size_t NpyReader::data_size() const {
  if (!owned_data_.empty()) {
    return owned_data_.size();
  }
  return mmap_.size;
}

NpyReader NpyReader::open(const std::filesystem::path& path, const char* label) {
#if !defined(_WIN32)
  auto mmap = detail::mmap_file(path);
  const auto parsed = parse_npy_f32_matrix(static_cast<const char*>(mmap.data), mmap.size);
  return NpyReader({}, std::move(mmap), label, parsed.dim, parsed.count, parsed.data_offset);
#else
  try {
    auto mmap = detail::mmap_file(path);
    const auto parsed = parse_npy_f32_matrix(static_cast<const char*>(mmap.data), mmap.size);
    return NpyReader({}, std::move(mmap), label, parsed.dim, parsed.count, parsed.data_offset);
  } catch (...) {
    auto data = read_file_owned(path);
    const auto parsed = parse_npy_f32_matrix(data.data(), data.size());
    return NpyReader(std::move(data), {}, label, parsed.dim, parsed.count, parsed.data_offset);
  }
#endif
}

DatasetMeta NpyReader::meta() const {
  return DatasetMeta{dim_, count_, label_};
}

std::optional<std::span<const float>> NpyReader::next_vector_span() {
  if (index_ >= count_) {
    return std::nullopt;
  }

  const std::size_t byte_offset = data_offset_ + index_ * dim_ * 4;
  const std::size_t end = byte_offset + dim_ * 4;
  if (end > data_size()) {
    throw Error("NPY data truncated at vector " + std::to_string(index_));
  }

  const auto* floats = reinterpret_cast<const float*>(data_ptr() + byte_offset);
  ++index_;
  return std::span<const float>(floats, dim_);
}

bool NpyReader::next_vector_into(std::span<float> out) {
  if (out.size() != dim_) {
    throw Error("buffer dimension mismatch: expected " + std::to_string(dim_) + ", got " +
                std::to_string(out.size()));
  }

  auto view = next_vector_span();
  if (!view) {
    return false;
  }

  std::memcpy(out.data(), view->data(), dim_ * sizeof(float));
  return true;
}

void write_npy_f32_matrix(const std::filesystem::path& path, std::size_t rows, std::size_t cols,
                          const float* data) {
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }

  std::ostringstream header;
  header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols
         << "), }";
  std::string header_str = header.str();
  const std::size_t padding = (16 - ((10 + header_str.size()) % 16)) % 16;
  header_str.append(padding, ' ');
  header_str.push_back('\n');

  std::ofstream file(path, std::ios::binary);
  if (!file) {
    throw Error("failed to create " + path.string());
  }

  static constexpr char kMagic[] = "\x93NUMPY";
  file.write(kMagic, 6);
  const char version[2] = {1, 0};
  file.write(version, 2);
  const std::uint16_t header_len = static_cast<std::uint16_t>(header_str.size());
  file.write(reinterpret_cast<const char*>(&header_len), 2);
  file.write(header_str.data(), static_cast<std::streamsize>(header_str.size()));
  file.write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(rows * cols * sizeof(float)));
}

}  // namespace vectorcache::datasets
