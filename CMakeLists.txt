cmake_minimum_required(VERSION 3.20)
project(vectorcache VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(VECTORCACHE_BUILD_GLOVE "Enable GloVe HDF5 reader" ON)
option(VECTORCACHE_BUILD_TOOLS "Build CLI tools" ON)
option(VECTORCACHE_BUILD_TESTS "Build tests" ON)
option(VECTORCACHE_FETCH_OPENAI "Download/convert OpenAI parquet shards (requires Arrow)" ON)

find_package(OpenMP REQUIRED)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-mavx512f -mavx512dq -mavx512bw -mfma)
elseif(MSVC)
  add_compile_options(/arch:AVX512)
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)

FetchContent_Declare(
  CLI11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.4.2
)
FetchContent_MakeAvailable(CLI11)

set(VECTORCACHE_SOURCES
  src/transform/fwht.cpp
  src/transform/normalize.cpp
  src/transform/srht.cpp
  src/quantize/quantize.cpp
  src/ingest/block.cpp
  src/ingest/store.cpp
  src/ingest/timing.cpp
  src/ingest/engine.cpp
  src/datasets/npy.cpp
)

if(VECTORCACHE_BUILD_GLOVE)
  list(APPEND VECTORCACHE_SOURCES src/datasets/hdf5.cpp)
endif()

add_library(vectorcache ${VECTORCACHE_SOURCES})
target_include_directories(vectorcache PUBLIC include)
target_compile_features(vectorcache PUBLIC cxx_std_20)
target_link_libraries(vectorcache PUBLIC OpenMP::OpenMP_CXX)

if(VECTORCACHE_BUILD_GLOVE)
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_compile_definitions(vectorcache PUBLIC VECTORCACHE_BUILD_GLOVE)
  target_include_directories(vectorcache PUBLIC ${HDF5_INCLUDE_DIRS})
  target_link_libraries(vectorcache PUBLIC ${HDF5_LIBRARIES})
endif()

if(VECTORCACHE_BUILD_TOOLS)
  find_package(CURL REQUIRED)

  target_sources(vectorcache PRIVATE src/datasets/datasets.cpp)
  target_link_libraries(vectorcache PUBLIC CURL::libcurl)

  if(VECTORCACHE_FETCH_OPENAI)
    find_package(Arrow CONFIG REQUIRED)
    find_package(Parquet CONFIG REQUIRED)
    target_compile_definitions(vectorcache PRIVATE VECTORCACHE_FETCH_OPENAI)

    if(TARGET Arrow::arrow_shared)
      target_link_libraries(vectorcache PUBLIC Arrow::arrow_shared Parquet::parquet_shared)
    elseif(TARGET Arrow::arrow_static)
      target_link_libraries(vectorcache PUBLIC Arrow::arrow_static Parquet::parquet_static)
    elseif(TARGET Arrow::arrow)
      target_link_libraries(vectorcache PUBLIC Arrow::arrow Parquet::parquet)
    else()
      message(FATAL_ERROR "Could not find Arrow CMake target")
    endif()
  endif()
endif()

if(VECTORCACHE_BUILD_TOOLS)
  function(add_vectorcache_tool name)
    add_executable(${name} tools/${name}.cpp)
    target_link_libraries(${name} PRIVATE vectorcache CLI11::CLI11)
  endfunction()

  add_vectorcache_tool(fetch-datasets)
  add_vectorcache_tool(ingest-sample)
  add_vectorcache_tool(ingest-bench)
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
    tests/test_fwht.cpp
    tests/test_normalize.cpp
    tests/test_srht.cpp
    tests/test_quantize.cpp
    tests/test_block.cpp
    tests/test_store.cpp
    tests/test_engine.cpp
    tests/test_timing.cpp
    tests/test_npy.cpp
  )

  if(VECTORCACHE_BUILD_GLOVE)
    target_sources(vectorcache_tests PRIVATE tests/test_hdf5.cpp)
  endif()

  target_link_libraries(vectorcache_tests PRIVATE vectorcache GTest::gtest)
  include(GoogleTest)
  gtest_discover_tests(vectorcache_tests)
endif()
