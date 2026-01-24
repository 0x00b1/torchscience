/**
 * CUDA backend implementation for window functions.
 *
 * This file provides CUDA implementations for parameterless window functions.
 * Each window function uses the device-agnostic kernel headers that contain
 * C10_HOST_DEVICE functions callable from both CPU and CUDA code.
 */

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

// Include kernel headers for parameterless windows
#include "../../kernel/signal_processing/window_function/hann.h"
#include "../../kernel/signal_processing/window_function/hamming.h"
#include "../../kernel/signal_processing/window_function/blackman.h"
#include "../../kernel/signal_processing/window_function/bartlett.h"
#include "../../kernel/signal_processing/window_function/cosine.h"
#include "../../kernel/signal_processing/window_function/nuttall.h"
#include "../../kernel/signal_processing/window_function/triangular.h"
#include "../../kernel/signal_processing/window_function/welch.h"
#include "../../kernel/signal_processing/window_function/parzen.h"
#include "../../kernel/signal_processing/window_function/blackman_harris.h"
#include "../../kernel/signal_processing/window_function/flat_top.h"
#include "../../kernel/signal_processing/window_function/sine.h"
#include "../../kernel/signal_processing/window_function/bartlett_hann.h"
#include "../../kernel/signal_processing/window_function/lanczos.h"

namespace torchscience::cuda::window_function {

namespace {

constexpr int BLOCK_SIZE = 256;

// =============================================================================
// Helper function for building tensor options
// =============================================================================

inline at::TensorOptions build_window_options(
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCUDA));
}

// =============================================================================
// CUDA kernels for parameterless windows
// =============================================================================

template<typename scalar_t>
__global__ void hann_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::hann<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void hamming_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::hamming<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void blackman_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::blackman<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void bartlett_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::bartlett<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void cosine_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::cosine<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void nuttall_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::nuttall<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void triangular_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::triangular<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void welch_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::welch<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void parzen_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::parzen<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void blackman_harris_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::blackman_harris<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void flat_top_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::flat_top<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void sine_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::sine<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void bartlett_hann_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::bartlett_hann<scalar_t>(i, n, periodic);
  }
}

template<typename scalar_t>
__global__ void lanczos_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::lanczos<scalar_t>(i, n, periodic);
  }
}

}  // anonymous namespace

// =============================================================================
// Macro for parameterless window implementation
// =============================================================================

#define TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(name)                            \
at::Tensor name##_window_impl(                                                  \
  int64_t n,                                                                    \
  bool periodic,                                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  TORCH_CHECK(n >= 0, #name "_window: n must be non-negative, got ", n);        \
                                                                                \
  auto options = build_window_options(dtype, layout, device);                   \
  auto output = at::empty({n}, options);                                        \
                                                                                \
  if (n == 0) {                                                                 \
    return output.requires_grad_(requires_grad);                                \
  }                                                                             \
                                                                                \
  c10::cuda::CUDAGuard device_guard(output.device());                           \
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;                         \
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();                       \
                                                                                \
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    output.scalar_type(),                                                       \
    #name "_window_cuda",                                                       \
    [&] {                                                                       \
      name##_window_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(        \
        output.data_ptr<scalar_t>(), n, periodic                                \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
  return output.requires_grad_(requires_grad);                                  \
}                                                                               \
                                                                                \
at::Tensor name##_window(                                                       \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, false, dtype, layout, device, requires_grad);    \
}                                                                               \
                                                                                \
at::Tensor periodic_##name##_window(                                            \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, true, dtype, layout, device, requires_grad);     \
}

// Generate implementations for all parameterless windows
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(hann)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(hamming)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(blackman)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(bartlett)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(cosine)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(nuttall)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(triangular)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(welch)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(parzen)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(blackman_harris)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(flat_top)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(sine)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(bartlett_hann)
TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW(lanczos)

#undef TORCHSCIENCE_CUDA_PARAMETERLESS_WINDOW

}  // namespace torchscience::cuda::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations for CUDA backend
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CUDA, m) {
  // Parameterless windows - symmetric
  m.impl("hann_window", torchscience::cuda::window_function::hann_window);
  m.impl("hamming_window", torchscience::cuda::window_function::hamming_window);
  m.impl("blackman_window", torchscience::cuda::window_function::blackman_window);
  m.impl("bartlett_window", torchscience::cuda::window_function::bartlett_window);
  m.impl("cosine_window", torchscience::cuda::window_function::cosine_window);
  m.impl("nuttall_window", torchscience::cuda::window_function::nuttall_window);
  m.impl("triangular_window", torchscience::cuda::window_function::triangular_window);
  m.impl("welch_window", torchscience::cuda::window_function::welch_window);
  m.impl("parzen_window", torchscience::cuda::window_function::parzen_window);
  m.impl("blackman_harris_window", torchscience::cuda::window_function::blackman_harris_window);
  m.impl("flat_top_window", torchscience::cuda::window_function::flat_top_window);
  m.impl("sine_window", torchscience::cuda::window_function::sine_window);
  m.impl("bartlett_hann_window", torchscience::cuda::window_function::bartlett_hann_window);
  m.impl("lanczos_window", torchscience::cuda::window_function::lanczos_window);

  // Parameterless windows - periodic
  m.impl("periodic_hann_window", torchscience::cuda::window_function::periodic_hann_window);
  m.impl("periodic_hamming_window", torchscience::cuda::window_function::periodic_hamming_window);
  m.impl("periodic_blackman_window", torchscience::cuda::window_function::periodic_blackman_window);
  m.impl("periodic_bartlett_window", torchscience::cuda::window_function::periodic_bartlett_window);
  m.impl("periodic_cosine_window", torchscience::cuda::window_function::periodic_cosine_window);
  m.impl("periodic_nuttall_window", torchscience::cuda::window_function::periodic_nuttall_window);
  m.impl("periodic_triangular_window", torchscience::cuda::window_function::periodic_triangular_window);
  m.impl("periodic_welch_window", torchscience::cuda::window_function::periodic_welch_window);
  m.impl("periodic_parzen_window", torchscience::cuda::window_function::periodic_parzen_window);
  m.impl("periodic_blackman_harris_window", torchscience::cuda::window_function::periodic_blackman_harris_window);
  m.impl("periodic_flat_top_window", torchscience::cuda::window_function::periodic_flat_top_window);
  m.impl("periodic_sine_window", torchscience::cuda::window_function::periodic_sine_window);
  m.impl("periodic_bartlett_hann_window", torchscience::cuda::window_function::periodic_bartlett_hann_window);
  m.impl("periodic_lanczos_window", torchscience::cuda::window_function::periodic_lanczos_window);
}
