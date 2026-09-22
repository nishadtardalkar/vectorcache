cmake_minimum_required(VERSION 3.20)
project(vectorcache VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VECTORCACHE_BUILD_GLOVE "Enable GloVe HDF5 reader" ON)
option(VECTORCACHE_BUILD_TOOLS "Build CLI tools" ON)
option(VECTORCACHE_BUILD_TESTS "Build GoogleTest suite" ON)
option(VECTORCACHE_OPENMP "Enable OpenMP parallelism" ON)
option(VECTORCACHE_FETCH_DATASETS "Build fetch-datasets (requires libcurl)" ON)
option(VECTORCACHE_FETCH_OPENAI "Download/convert OpenAI parquet shards (requires Arrow)" ON)

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

set(VECTORCACHE_SOURCES
  src/chacha8.cpp
  src/transform/rotation.cpp
  src/quantize/codebook.cpp
  src/encode/encode.cpp
  src/pack/pack.cpp
  src/search/search.cpp
  src/search/search_avx2.cpp
  src/search/search_vnni.cpp
  src/index.cpp
  src/datasets/npy.cpp
  src/datasets/datasets.cpp
)

if(VECTORCACHE_BUILD_GLOVE)
  list(APPEND VECTORCACHE_SOURCES src/datasets/hdf5.cpp)
endif()

add_library(vectorcache ${VECTORCACHE_SOURCES})
target_include_directories(vectorcache PUBLIC include)
target_compile_features(vectorcache PUBLIC cxx_std_20)

if(MSVC)
  target_compile_options(vectorcache PRIVATE /W4 /arch:AVX2)
  set_source_files_properties(src/search/search_vnni.cpp PROPERTIES
    COMPILE_OPTIONS "/arch:AVX512")
else()
  target_compile_options(vectorcache PRIVATE
    -Wall -Wextra
    -mavx2 -mfma
  )
  # Compile SIMD search TUs with broader ISA; runtime dispatch selects path.
  # Use COMPILE_OPTIONS (not COMPILE_FLAGS) so flags survive modern CMake generators.
  set_source_files_properties(src/search/search_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/search/search_vnni.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vbmi;-mavx512vnni;-mfma")
endif()

if(VECTORCACHE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(vectorcache PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(vectorcache PUBLIC VECTORCACHE_OPENMP)
  endif()
endif()

if(VECTORCACHE_BUILD_GLOVE)
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_compile_definitions(vectorcache PUBLIC VECTORCACHE_BUILD_GLOVE)
  target_include_directories(vectorcache PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(vectorcache PUBLIC ${HDF5_LIBRARIES})
endif()

function(vectorcache_link_parquet_libstdcxx link_target parquet_target)
  if(NOT TARGET ${parquet_target})
    return()
  endif()
  foreach(_prop IMPORTED_LOCATION IMPORTED_LOCATION_RELEASE IMPORTED_LOCATION_DEBUG)
    get_target_property(_parquet_loc ${parquet_target} ${_prop})
    if(_parquet_loc)
      break()
    endif()
  endforeach()
  if(NOT _parquet_loc)
    return()
  endif()
  get_filename_component(_parquet_libdir "${_parquet_loc}" DIRECTORY)
  find_library(VECTORCACHE_PARQUET_LIBSTDCXX stdc++ PATHS "${_parquet_libdir}" NO_DEFAULT_PATH)
  if(VECTORCACHE_PARQUET_LIBSTDCXX)
    target_link_libraries(${link_target} PUBLIC "${VECTORCACHE_PARQUET_LIBSTDCXX}")
    target_link_options(${link_target} PUBLIC "LINKER:-rpath,${_parquet_libdir}")
  endif()
endfunction()

if(VECTORCACHE_BUILD_TOOLS AND VECTORCACHE_FETCH_DATASETS)
  find_package(CURL REQUIRED)

  add_library(vectorcache_fetch STATIC src/datasets/fetch.cpp)
  target_link_libraries(vectorcache_fetch PUBLIC vectorcache CURL::libcurl)
  target_include_directories(vectorcache_fetch PUBLIC include)

  if(VECTORCACHE_FETCH_OPENAI)
    find_package(Arrow CONFIG REQUIRED)
    find_package(Parquet CONFIG REQUIRED)
    target_compile_definitions(vectorcache_fetch PRIVATE VECTORCACHE_FETCH_OPENAI)

    if(TARGET Arrow::arrow_shared)
      target_link_libraries(vectorcache_fetch PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
      vectorcache_link_parquet_libstdcxx(vectorcache_fetch Parquet::parquet_shared)
    elseif(TARGET Arrow::arrow_static)
      target_link_libraries(vectorcache_fetch PUBLIC Arrow::arrow_static Parquet::parquet_static)
    elseif(TARGET Arrow::arrow)
      target_link_libraries(vectorcache_fetch PUBLIC Arrow::arrow Parquet::parquet)
    else()
      message(FATAL_ERROR "Could not find Arrow CMake target")
    endif()
  endif()

  add_executable(fetch-datasets tools/fetch-datasets.cpp)
  target_link_libraries(fetch-datasets PRIVATE vectorcache vectorcache_fetch CLI11::CLI11)
endif()

if(VECTORCACHE_BUILD_TOOLS)
  add_executable(query-bench tools/query-bench.cpp)
  target_link_libraries(query-bench PRIVATE vectorcache CLI11::CLI11)
endif()

if(VECTORCACHE_BUILD_TESTS)
  enable_testing()
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  add_executable(vectorcache_tests
    tests/test_main.cpp
    tests/test_rotation.cpp
    tests/test_codebook.cpp
    tests/test_pack.cpp
    tests/test_index.cpp
    tests/test_search_vnni.cpp
  )
  target_link_libraries(vectorcache_tests PRIVATE vectorcache GTest::gtest)
  include(GoogleTest)
  gtest_discover_tests(vectorcache_tests)
endif()
