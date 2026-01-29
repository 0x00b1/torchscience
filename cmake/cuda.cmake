# cmake/cuda.cmake
# CUDA backend configuration - only included when TORCHSCIENCE_ENABLE_CUDA=ON

enable_language(CUDA)

# Find CUDA toolkit
find_package(CUDAToolkit 12.0 REQUIRED)
message(STATUS "Found CUDA Toolkit: ${CUDAToolkit_VERSION}")

# CUDA sources
set(TORCHSCIENCE_CUDA_SOURCES
  src/torchscience/csrc/cuda/special_functions/chebyshev_polynomial_t.cu
  src/torchscience/csrc/cuda/special_functions/gamma.cu
  src/torchscience/csrc/cuda/signal_processing/filter.cu
  src/torchscience/csrc/cuda/signal_processing/window_functions.cu
  src/torchscience/csrc/cuda/statistics/descriptive/kurtosis.cu
  src/torchscience/csrc/cuda/integral_transform/hilbert_transform.cu
  src/torchscience/csrc/cuda/integral_transform/inverse_hilbert_transform.cu
  src/torchscience/csrc/cuda/geometry/intersection/ray_plane.cu
  src/torchscience/csrc/cuda/geometry/intersection/ray_sphere.cu
  src/torchscience/csrc/cuda/geometry/intersection/ray_triangle.cu
  src/torchscience/csrc/cuda/geometry/intersection/ray_aabb.cu
  src/torchscience/csrc/cuda/transform/discrete_wavelet_transform.cu
  src/torchscience/csrc/cuda/transform/inverse_discrete_wavelet_transform.cu
)

# Add CUDA sources to target
target_sources(_csrc PRIVATE ${TORCHSCIENCE_CUDA_SOURCES})

# CUDA-specific includes
target_include_directories(_csrc PRIVATE ${CUDAToolkit_INCLUDE_DIRS})

# CUDA-specific linking
target_link_libraries(_csrc PRIVATE CUDA::cudart)

# Compile definition to enable CUDA code paths
target_compile_definitions(_csrc PRIVATE TORCHSCIENCE_CUDA)

message(STATUS "CUDA backend enabled with ${CUDAToolkit_VERSION}")
