cmake_minimum_required(VERSION 3.18)

if(NOT DEFINED SKBUILD_PROJECT_NAME)
  set(SKBUILD_PROJECT_NAME "torchscience")
endif()

project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

# Build options
option(TORCHSCIENCE_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(TORCHSCIENCE_BUILD_TESTS "Build tests" OFF)

if((TORCHSCIENCE_BUILD_BENCHMARKS OR TORCHSCIENCE_BUILD_TESTS) AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Core dependencies
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(Torch REQUIRED)

# Backend detection (sets TORCHSCIENCE_ENABLE_CUDA, TORCHSCIENCE_ENABLE_MPS)
include(cmake/backend_detection.cmake)

# Base sources
set(TORCHSCIENCE_SOURCES src/torchscience/csrc/torchscience.cpp)

# Create library target
add_library(_csrc MODULE ${TORCHSCIENCE_SOURCES})

# Include backend configurations
include(cmake/cpu.cmake)
include(cmake/embree.cmake)

if(TORCHSCIENCE_ENABLE_CUDA)
  include(cmake/cuda.cmake)
endif()

if(TORCHSCIENCE_ENABLE_MPS)
  include(cmake/mps.cmake)
endif()

# Common configuration
target_include_directories(_csrc PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_csrc PRIVATE ${TORCH_LIBRARIES} Python::Module)

set_target_properties(_csrc PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  CXX_STANDARD 17
  INSTALL_RPATH_USE_LINK_PATH TRUE
  PREFIX ""
)

if(APPLE)
  set_target_properties(_csrc PROPERTIES
    INSTALL_RPATH "@loader_path;@loader_path/../torch/lib"
    SUFFIX ".so"
  )
endif()

# Install
if(SKBUILD)
  install(TARGETS _csrc DESTINATION ${SKBUILD_PROJECT_NAME})
endif()

# Benchmarks and tests
if(TORCHSCIENCE_BUILD_BENCHMARKS OR TORCHSCIENCE_BUILD_TESTS)
  include(FetchContent)
endif()

if(TORCHSCIENCE_BUILD_BENCHMARKS)
  FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
  add_subdirectory(benchmark)
endif()

if(TORCHSCIENCE_BUILD_TESTS)
  FetchContent_Declare(googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  enable_testing()
  add_subdirectory(test)
endif()