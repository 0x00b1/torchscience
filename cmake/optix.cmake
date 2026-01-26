# cmake/optix.cmake
# OptiX backend configuration - only included when CUDA is enabled
# OptiX provides GPU-accelerated BVH ray tracing using NVIDIA RT cores

# Find OptiX SDK (header-only)
find_path(OptiX_INCLUDE_DIR
  NAMES optix.h
  HINTS
    ${OptiX_INSTALL_DIR}/include
    $ENV{OptiX_INSTALL_DIR}/include
    /usr/local/include
    /opt/optix/include
  PATH_SUFFIXES include
)

if(NOT OptiX_INCLUDE_DIR)
  message(STATUS "OptiX SDK not found - OptiX BVH features disabled")
  message(STATUS "  Set OptiX_INSTALL_DIR to enable OptiX support")
  return()
endif()

# Extract OptiX version
file(STRINGS "${OptiX_INCLUDE_DIR}/optix.h" _optix_version_line
  REGEX "#define OPTIX_VERSION")
if(_optix_version_line)
  string(REGEX MATCH "[0-9]+" _optix_version_raw "${_optix_version_line}")
  math(EXPR OptiX_VERSION_MAJOR "${_optix_version_raw} / 10000")
  math(EXPR OptiX_VERSION_MINOR "(${_optix_version_raw} % 10000) / 100")
  math(EXPR OptiX_VERSION_MICRO "${_optix_version_raw} % 100")
  set(OptiX_VERSION "${OptiX_VERSION_MAJOR}.${OptiX_VERSION_MINOR}.${OptiX_VERSION_MICRO}")
  message(STATUS "Found OptiX: ${OptiX_VERSION} at ${OptiX_INCLUDE_DIR}")
else()
  message(STATUS "Found OptiX at ${OptiX_INCLUDE_DIR} (version unknown)")
endif()

# Check minimum version (OptiX 7.0+)
if(DEFINED OptiX_VERSION_MAJOR AND OptiX_VERSION_MAJOR LESS 7)
  message(WARNING "OptiX ${OptiX_VERSION} found but >= 7.0 required - OptiX disabled")
  return()
endif()

# Find bin2c for embedding PTX
find_program(BIN2C bin2c
  HINTS ${CUDAToolkit_BIN_DIR}
)

if(NOT BIN2C)
  message(STATUS "bin2c not found - OptiX disabled (needed to embed PTX)")
  return()
endif()

# Compile OptiX device programs to PTX
set(OPTIX_PROGRAMS_SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/src/torchscience/csrc/optix/programs.cu)

set(OPTIX_PTX_FILE ${CMAKE_BINARY_DIR}/torchscience_optix_programs.ptx)
set(OPTIX_PTX_HEADER ${CMAKE_BINARY_DIR}/generated/torchscience_optix_programs_ptx.h)

# Create output directory
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/generated)

# Compile .cu to .ptx using nvcc
add_custom_command(
  OUTPUT ${OPTIX_PTX_FILE}
  COMMAND ${CMAKE_CUDA_COMPILER}
    -ptx
    -std=c++17
    --use_fast_math
    -I${OptiX_INCLUDE_DIR}
    -I${CUDAToolkit_INCLUDE_DIRS}
    -I${CMAKE_CURRENT_SOURCE_DIR}/src/torchscience/csrc
    ${OPTIX_PROGRAMS_SOURCE}
    -o ${OPTIX_PTX_FILE}
  DEPENDS ${OPTIX_PROGRAMS_SOURCE}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/torchscience/csrc/optix/launch_params.h
  COMMENT "Compiling OptiX device programs to PTX"
  VERBATIM
)

# Embed PTX as C string constant
add_custom_command(
  OUTPUT ${OPTIX_PTX_HEADER}
  COMMAND ${BIN2C}
    --padd 0
    --type char
    --name torchscience_optix_programs_ptx
    ${OPTIX_PTX_FILE}
    > ${OPTIX_PTX_HEADER}
  DEPENDS ${OPTIX_PTX_FILE}
  COMMENT "Embedding OptiX PTX as C string"
  VERBATIM
)

# Custom target to ensure PTX is built before main library
add_custom_target(optix_ptx_embed DEPENDS ${OPTIX_PTX_HEADER})
add_dependencies(_csrc optix_ptx_embed)

# Add OptiX configuration to main target
target_include_directories(_csrc PRIVATE
  ${OptiX_INCLUDE_DIR}
  ${CMAKE_BINARY_DIR}/generated
)

target_link_libraries(_csrc PRIVATE CUDA::cuda_driver)

target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_OPTIX)

message(STATUS "OptiX backend enabled with ${OptiX_VERSION}")
