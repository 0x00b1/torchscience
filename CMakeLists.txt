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

# Get Torch installation path from the Python environment to avoid finding system Torch
execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import torch; import os; print(os.path.dirname(torch.__file__))"
  OUTPUT_VARIABLE TORCH_INSTALL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _torch_result
)
if(_torch_result EQUAL 0 AND TORCH_INSTALL_PREFIX)
  # Set TORCH_INSTALL_PREFIX so TorchConfig.cmake uses the correct torch
  set(ENV{TORCH_INSTALL_PREFIX} ${TORCH_INSTALL_PREFIX})
  # Prepend to CMAKE_LIBRARY_PATH so find_library finds venv's torch libs first
  list(PREPEND CMAKE_LIBRARY_PATH "${TORCH_INSTALL_PREFIX}/lib")
  list(PREPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}/share/cmake")
endif()

find_package(Torch REQUIRED)

# Backend detection (sets TORCHSCIENCE_ENABLE_CUDA, TORCHSCIENCE_ENABLE_MPS)
include(cmake/backend_detection.cmake)

# Base sources
set(TORCHSCIENCE_SOURCES
  src/torchscience/csrc/torchscience.cpp
  src/torchscience/csrc/special_functions.cpp
)

# Create library target
add_library(_csrc MODULE ${TORCHSCIENCE_SOURCES})

# Include backend configurations
include(cmake/cpu.cmake)

if(TORCHSCIENCE_ENABLE_CUDA)
  include(cmake/cuda.cmake)
endif()

if(TORCHSCIENCE_ENABLE_MPS)
  include(cmake/mps.cmake)
endif()

# Common configuration
# IMPORTANT: Add Torch includes BEFORE other system includes to avoid conflicts
# with system-installed Torch (e.g., from Homebrew)
target_include_directories(_csrc BEFORE PRIVATE ${TORCH_INCLUDE_DIRS})
target_include_directories(_csrc PRIVATE ${Python_INCLUDE_DIRS})
target_link_libraries(_csrc PRIVATE ${TORCH_LIBRARIES} Python::Module)

set_target_properties(_csrc PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  CXX_STANDARD 17
  # Don't add link-time library paths to rpath (avoids temp build env paths)
  INSTALL_RPATH_USE_LINK_PATH FALSE
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