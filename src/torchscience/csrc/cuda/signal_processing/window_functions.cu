/**
 * CUDA backend implementation for window functions.
 *
 * This file provides CUDA implementations for window functions including:
 * - Parameterless windows (hann, hamming, blackman, etc.)
 * - Single-parameter windows (gaussian, kaiser, tukey, etc.)
 * - Two-parameter windows (generalized_normal, planck_bessel)
 * - Array-parameter windows (general_cosine)
 *
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

// Include kernel headers for parametric windows
#include "../../kernel/signal_processing/window_function/gaussian.h"
#include "../../kernel/signal_processing/window_function/general_hamming.h"
#include "../../kernel/signal_processing/window_function/tukey.h"
#include "../../kernel/signal_processing/window_function/exponential.h"
#include "../../kernel/signal_processing/window_function/hann_poisson.h"
#include "../../kernel/signal_processing/window_function/kaiser.h"
#include "../../kernel/signal_processing/window_function/planck_taper.h"
#include "../../kernel/signal_processing/window_function/generalized_normal.h"
#include "../../kernel/signal_processing/window_function/planck_bessel.h"
#include "../../kernel/signal_processing/window_function/general_cosine.h"

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

// =============================================================================
// CUDA kernels for parametric windows (single parameter)
// =============================================================================

template<typename scalar_t>
__global__ void gaussian_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t std_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::gaussian<scalar_t>(i, n, std_val, periodic);
  }
}

template<typename scalar_t>
__global__ void general_hamming_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t alpha_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::general_hamming<scalar_t>(i, n, alpha_val, periodic);
  }
}

template<typename scalar_t>
__global__ void tukey_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t alpha_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::tukey<scalar_t>(i, n, alpha_val, periodic);
  }
}

template<typename scalar_t>
__global__ void exponential_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t tau_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::exponential<scalar_t>(i, n, tau_val, periodic);
  }
}

template<typename scalar_t>
__global__ void hann_poisson_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t alpha_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::hann_poisson<scalar_t>(i, n, alpha_val, periodic);
  }
}

template<typename scalar_t>
__global__ void kaiser_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t beta_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::kaiser<scalar_t>(i, n, beta_val, periodic);
  }
}

template<typename scalar_t>
__global__ void planck_taper_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t epsilon_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::planck_taper<scalar_t>(i, n, epsilon_val, periodic);
  }
}

// =============================================================================
// CUDA kernels for parametric windows (two parameters)
// =============================================================================

template<typename scalar_t>
__global__ void generalized_normal_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t p_val,
  scalar_t sigma_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::generalized_normal<scalar_t>(i, n, p_val, sigma_val, periodic);
  }
}

template<typename scalar_t>
__global__ void planck_bessel_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  scalar_t epsilon_val,
  scalar_t beta_val,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::planck_bessel<scalar_t>(i, n, epsilon_val, beta_val, periodic);
  }
}

// =============================================================================
// CUDA kernel for general cosine window (array parameter)
// =============================================================================

template<typename scalar_t>
__global__ void general_cosine_window_kernel(
  scalar_t* __restrict__ output,
  int64_t n,
  const scalar_t* __restrict__ coeffs,
  int64_t num_coeffs,
  bool periodic
) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = kernel::window_function::general_cosine<scalar_t>(i, n, coeffs, num_coeffs, periodic);
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

// =============================================================================
// Macro for single-parameter window implementation
// =============================================================================

#define TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(name, param_name)                 \
at::Tensor name##_window_impl(                                                  \
  int64_t n,                                                                    \
  const at::Tensor& param_input,                                                \
  bool periodic,                                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  TORCH_CHECK(n >= 0, #name "_window: n must be non-negative, got ", n);        \
  TORCH_CHECK(param_input.dim() == 0,                                           \
    #name "_window: " #param_name " must be a scalar tensor");                  \
                                                                                \
  auto out_dtype = dtype.value_or(param_input.scalar_type());                   \
  auto options = at::TensorOptions()                                            \
    .dtype(out_dtype)                                                           \
    .layout(layout.value_or(at::kStrided))                                      \
    .device(device.value_or(at::kCUDA));                                        \
                                                                                \
  auto output = at::empty({n}, options);                                        \
                                                                                \
  if (n == 0) {                                                                 \
    return output;                                                              \
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
      scalar_t param_val = param_input.item<scalar_t>();                        \
      name##_window_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(        \
        output.data_ptr<scalar_t>(), n, param_val, periodic                     \
      );                                                                        \
    }                                                                           \
  );                                                                            \
                                                                                \
  C10_CUDA_KERNEL_LAUNCH_CHECK();                                               \
  return output;                                                                \
}                                                                               \
                                                                                \
at::Tensor name##_window(                                                       \
  int64_t n,                                                                    \
  const at::Tensor& param_input,                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  return name##_window_impl(n, param_input, false, dtype, layout, device);      \
}                                                                               \
                                                                                \
at::Tensor periodic_##name##_window(                                            \
  int64_t n,                                                                    \
  const at::Tensor& param_input,                                                \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device                                              \
) {                                                                             \
  return name##_window_impl(n, param_input, true, dtype, layout, device);       \
}

// Generate implementations for single-parameter windows
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(gaussian, std)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(general_hamming, alpha)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(tukey, alpha)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(exponential, tau)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(hann_poisson, alpha)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(kaiser, beta)
TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW(planck_taper, epsilon)

#undef TORCHSCIENCE_CUDA_SINGLE_PARAM_WINDOW

// =============================================================================
// Two-parameter window implementations
// =============================================================================

at::Tensor generalized_normal_window_impl(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "generalized_normal_window: n must be non-negative, got ", n);
  TORCH_CHECK(p_input.dim() == 0, "generalized_normal_window: p must be a scalar tensor");
  TORCH_CHECK(sigma_input.dim() == 0, "generalized_normal_window: sigma must be a scalar tensor");

  auto promoted = at::result_type(p_input, sigma_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCUDA));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  c10::cuda::CUDAGuard device_guard(output.device());
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    output.scalar_type(),
    "generalized_normal_window_cuda",
    [&] {
      scalar_t p_val = p_input.item<scalar_t>();
      scalar_t sigma_val = sigma_input.item<scalar_t>();
      generalized_normal_window_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
        output.data_ptr<scalar_t>(), n, p_val, sigma_val, periodic
      );
    }
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_normal_window_impl(n, p_input, sigma_input, false, dtype, layout, device);
}

at::Tensor periodic_generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_normal_window_impl(n, p_input, sigma_input, true, dtype, layout, device);
}

at::Tensor planck_bessel_window_impl(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "planck_bessel_window: n must be non-negative, got ", n);
  TORCH_CHECK(epsilon_input.dim() == 0, "planck_bessel_window: epsilon must be a scalar tensor");
  TORCH_CHECK(beta_input.dim() == 0, "planck_bessel_window: beta must be a scalar tensor");

  auto promoted = at::result_type(epsilon_input, beta_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCUDA));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  c10::cuda::CUDAGuard device_guard(output.device());
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    output.scalar_type(),
    "planck_bessel_window_cuda",
    [&] {
      scalar_t epsilon_val = epsilon_input.item<scalar_t>();
      scalar_t beta_val = beta_input.item<scalar_t>();
      planck_bessel_window_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
        output.data_ptr<scalar_t>(), n, epsilon_val, beta_val, periodic
      );
    }
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_bessel_window_impl(n, epsilon_input, beta_input, false, dtype, layout, device);
}

at::Tensor periodic_planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_bessel_window_impl(n, epsilon_input, beta_input, true, dtype, layout, device);
}

// =============================================================================
// General cosine window implementation (array parameter)
// =============================================================================

at::Tensor general_cosine_window_impl(
  int64_t n,
  const at::Tensor& coeffs_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "general_cosine_window: n must be non-negative, got ", n);
  TORCH_CHECK(coeffs_input.dim() == 1, "general_cosine_window: coeffs must be a 1-D tensor");
  TORCH_CHECK(coeffs_input.size(0) > 0, "general_cosine_window: coeffs must have at least one element");

  auto out_dtype = dtype.value_or(coeffs_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCUDA));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  int64_t num_coeffs = coeffs_input.size(0);

  c10::cuda::CUDAGuard device_guard(output.device());
  const int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16,
    at::kHalf,
    output.scalar_type(),
    "general_cosine_window_cuda",
    [&] {
      // Ensure coeffs are on CUDA and contiguous
      auto coeffs_cuda = coeffs_input.to(output.options()).contiguous();
      general_cosine_window_kernel<scalar_t><<<blocks, BLOCK_SIZE, 0, stream>>>(
        output.data_ptr<scalar_t>(), n, coeffs_cuda.data_ptr<scalar_t>(), num_coeffs, periodic
      );
    }
  );

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, false, dtype, layout, device);
}

at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, true, dtype, layout, device);
}

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

  // Single-parameter windows - symmetric
  m.impl("gaussian_window", torchscience::cuda::window_function::gaussian_window);
  m.impl("general_hamming_window", torchscience::cuda::window_function::general_hamming_window);
  m.impl("tukey_window", torchscience::cuda::window_function::tukey_window);
  m.impl("exponential_window", torchscience::cuda::window_function::exponential_window);
  m.impl("hann_poisson_window", torchscience::cuda::window_function::hann_poisson_window);
  m.impl("kaiser_window", torchscience::cuda::window_function::kaiser_window);
  m.impl("planck_taper_window", torchscience::cuda::window_function::planck_taper_window);
  m.impl("general_cosine_window", torchscience::cuda::window_function::general_cosine_window);

  // Single-parameter windows - periodic
  m.impl("periodic_gaussian_window", torchscience::cuda::window_function::periodic_gaussian_window);
  m.impl("periodic_general_hamming_window", torchscience::cuda::window_function::periodic_general_hamming_window);
  m.impl("periodic_tukey_window", torchscience::cuda::window_function::periodic_tukey_window);
  m.impl("periodic_exponential_window", torchscience::cuda::window_function::periodic_exponential_window);
  m.impl("periodic_hann_poisson_window", torchscience::cuda::window_function::periodic_hann_poisson_window);
  m.impl("periodic_kaiser_window", torchscience::cuda::window_function::periodic_kaiser_window);
  m.impl("periodic_planck_taper_window", torchscience::cuda::window_function::periodic_planck_taper_window);
  m.impl("periodic_general_cosine_window", torchscience::cuda::window_function::periodic_general_cosine_window);

  // Two-parameter windows - symmetric
  m.impl("generalized_normal_window", torchscience::cuda::window_function::generalized_normal_window);
  m.impl("planck_bessel_window", torchscience::cuda::window_function::planck_bessel_window);

  // Two-parameter windows - periodic
  m.impl("periodic_generalized_normal_window", torchscience::cuda::window_function::periodic_generalized_normal_window);
  m.impl("periodic_planck_bessel_window", torchscience::cuda::window_function::periodic_planck_bessel_window);
}
