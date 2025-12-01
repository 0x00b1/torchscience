#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include "../../kernel/signal_processing/window_function/hann.h"
#include "../../kernel/signal_processing/window_function/hamming.h"
#include "../../kernel/signal_processing/window_function/blackman.h"
#include "../../kernel/signal_processing/window_function/bartlett.h"
#include "../../kernel/signal_processing/window_function/cosine.h"
#include "../../kernel/signal_processing/window_function/nuttall.h"
#include "../../kernel/signal_processing/window_function/gaussian.h"
#include "../../kernel/signal_processing/window_function/general_hamming.h"
#include "../../kernel/signal_processing/window_function/general_cosine.h"
#include "../../kernel/signal_processing/window_function/triangular.h"
#include "../../kernel/signal_processing/window_function/welch.h"
#include "../../kernel/signal_processing/window_function/parzen.h"
#include "../../kernel/signal_processing/window_function/blackman_harris.h"
#include "../../kernel/signal_processing/window_function/flat_top.h"
#include "../../kernel/signal_processing/window_function/sine.h"
#include "../../kernel/signal_processing/window_function/bartlett_hann.h"
#include "../../kernel/signal_processing/window_function/lanczos.h"
#include "../../kernel/signal_processing/window_function/tukey.h"
#include "../../kernel/signal_processing/window_function/exponential.h"
#include "../../kernel/signal_processing/window_function/hann_poisson.h"
#include "../../kernel/signal_processing/window_function/generalized_normal.h"
#include "../../kernel/signal_processing/window_function/kaiser.h"
#include "../../kernel/signal_processing/window_function/planck_taper.h"
#include "../../kernel/signal_processing/window_function/planck_bessel.h"

namespace torchscience::cpu::window_function {

namespace {

inline at::TensorOptions build_window_options(
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::kCPU));
}

}  // anonymous namespace

// =============================================================================
// Parameterless windows macro
// =============================================================================

#define TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(name)                          \
inline at::Tensor name##_window_impl(                                           \
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
  AT_DISPATCH_FLOATING_TYPES_AND2(                                              \
    at::kBFloat16,                                                              \
    at::kHalf,                                                                  \
    output.scalar_type(),                                                       \
    #name "_window",                                                            \
    [&] {                                                                       \
      auto* out_ptr = output.data_ptr<scalar_t>();                              \
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {            \
        for (int64_t i = begin; i < end; ++i) {                                 \
          out_ptr[i] = kernel::window_function::name<scalar_t>(i, n, periodic); \
        }                                                                       \
      });                                                                       \
    }                                                                           \
  );                                                                            \
                                                                                \
  return output.requires_grad_(requires_grad);                                  \
}                                                                               \
                                                                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, false, dtype, layout, device, requires_grad);    \
}                                                                               \
                                                                                \
inline at::Tensor periodic_##name##_window(                                     \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  return name##_window_impl(n, true, dtype, layout, device, requires_grad);     \
}

TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(hann)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(hamming)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(blackman)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(bartlett)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(cosine)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(nuttall)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(triangular)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(welch)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(parzen)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(blackman_harris)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(flat_top)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(sine)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(bartlett_hann)
TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW(lanczos)

#undef TORCHSCIENCE_DEFINE_PARAMETERLESS_WINDOW

// =============================================================================
// Gaussian window
// =============================================================================

inline at::Tensor gaussian_window_impl(
  int64_t n,
  const at::Tensor& std_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "gaussian_window: n must be non-negative, got ", n);
  TORCH_CHECK(std_input.dim() == 0, "gaussian_window: std must be a scalar tensor");

  auto out_dtype = dtype.value_or(std_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(std_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "gaussian_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t std_val = std_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::gaussian<scalar_t>(i, n, std_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window_impl(n, std_input, false, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window_impl(n, std_input, true, dtype, layout, device);
}

inline at::Tensor gaussian_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input,
  bool periodic
) {
  auto grad_std = at::zeros_like(std_input);

  if (n == 0) {
    return grad_std;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "gaussian_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t std_val = std_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::gaussian_backward<scalar_t>(
          grad_out_ptr[i], i, n, std_val, periodic, out_ptr[i]
        );
      }

      grad_std.fill_(accum);
    }
  );

  return grad_std;
}

inline at::Tensor gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward_impl(grad_output, output, n, std_input, false);
}

inline at::Tensor periodic_gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward_impl(grad_output, output, n, std_input, true);
}

// =============================================================================
// General Hamming window
// =============================================================================

inline at::Tensor general_hamming_window_impl(
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "general_hamming_window: n must be non-negative, got ", n);
  TORCH_CHECK(alpha_input.dim() == 0, "general_hamming_window: alpha must be a scalar tensor");

  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(alpha_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "general_hamming_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::general_hamming<scalar_t>(i, n, alpha_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window_impl(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window_impl(n, alpha_input, true, dtype, layout, device);
}

inline at::Tensor general_hamming_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic
) {
  auto grad_alpha = at::zeros_like(alpha_input);

  if (n == 0) {
    return grad_alpha;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "general_hamming_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::general_hamming_backward<scalar_t>(
          grad_out_ptr[i], i, n, alpha_val, periodic, out_ptr[i]
        );
      }

      grad_alpha.fill_(accum);
    }
  );

  return grad_alpha;
}

inline at::Tensor general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward_impl(grad_output, output, n, alpha_input, false);
}

inline at::Tensor periodic_general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward_impl(grad_output, output, n, alpha_input, true);
}

// =============================================================================
// General Cosine window
// =============================================================================

inline at::Tensor general_cosine_window_impl(
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
    .device(device.value_or(coeffs_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  int64_t num_coeffs = coeffs_input.size(0);

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "general_cosine_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      auto coeffs_contig = coeffs_input.to(output.scalar_type()).contiguous();
      auto* coeffs_ptr = coeffs_contig.data_ptr<scalar_t>();

      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::general_cosine<scalar_t>(
            i, n, coeffs_ptr, num_coeffs, periodic
          );
        }
      });
    }
  );

  return output;
}

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, false, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window_impl(n, coeffs_input, true, dtype, layout, device);
}

inline at::Tensor general_cosine_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input,
  bool periodic
) {
  (void)output;
  int64_t num_coeffs = coeffs_input.size(0);
  auto grad_coeffs = at::zeros_like(coeffs_input);

  if (n == 0) {
    return grad_coeffs;
  }

  // Ensure contiguous tensor for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "general_cosine_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto grad_coeffs_accessor = grad_coeffs.accessor<scalar_t, 1>();

      for (int64_t j = 0; j < num_coeffs; ++j) {
        scalar_t accum = scalar_t(0);
        for (int64_t i = 0; i < n; ++i) {
          accum += kernel::window_function::general_cosine_backward_coeff<scalar_t>(
            grad_out_ptr[i], i, n, j, periodic
          );
        }
        grad_coeffs_accessor[j] = accum;
      }
    }
  );

  return grad_coeffs;
}

inline at::Tensor general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward_impl(grad_output, output, n, coeffs_input, false);
}

inline at::Tensor periodic_general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward_impl(grad_output, output, n, coeffs_input, true);
}

// =============================================================================
// Tukey window
// =============================================================================

inline at::Tensor tukey_window_impl(
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "tukey_window: n must be non-negative, got ", n);
  TORCH_CHECK(alpha_input.dim() == 0, "tukey_window: alpha must be a scalar tensor");

  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(alpha_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "tukey_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::tukey<scalar_t>(i, n, alpha_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return tukey_window_impl(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return tukey_window_impl(n, alpha_input, true, dtype, layout, device);
}

inline at::Tensor tukey_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic
) {
  auto grad_alpha = at::zeros_like(alpha_input);

  if (n == 0) {
    return grad_alpha;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "tukey_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::tukey_backward<scalar_t>(
          grad_out_ptr[i], i, n, alpha_val, periodic, out_ptr[i]
        );
      }

      grad_alpha.fill_(accum);
    }
  );

  return grad_alpha;
}

inline at::Tensor tukey_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return tukey_window_backward_impl(grad_output, output, n, alpha_input, false);
}

inline at::Tensor periodic_tukey_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return tukey_window_backward_impl(grad_output, output, n, alpha_input, true);
}

// =============================================================================
// Exponential window
// =============================================================================

inline at::Tensor exponential_window_impl(
  int64_t n,
  const at::Tensor& tau_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "exponential_window: n must be non-negative, got ", n);
  TORCH_CHECK(tau_input.dim() == 0, "exponential_window: tau must be a scalar tensor");
  TORCH_CHECK(!tau_input.is_complex(), "exponential_window: tau must be real-valued");

  auto out_dtype = dtype.value_or(tau_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(tau_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "exponential_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t tau_val = tau_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::exponential<scalar_t>(i, n, tau_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return exponential_window_impl(n, tau_input, false, dtype, layout, device);
}

inline at::Tensor periodic_exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return exponential_window_impl(n, tau_input, true, dtype, layout, device);
}

inline at::Tensor exponential_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& tau_input,
  bool periodic
) {
  auto grad_tau = at::zeros_like(tau_input);

  if (n == 0) {
    return grad_tau;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "exponential_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t tau_val = tau_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      // Serial accumulation (consistent with existing gaussian_window_backward)
      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::exponential_backward<scalar_t>(
          grad_out_ptr[i], i, n, tau_val, periodic, out_ptr[i]
        );
      }

      grad_tau.fill_(accum);
    }
  );

  return grad_tau;
}

inline at::Tensor exponential_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& tau_input
) {
  return exponential_window_backward_impl(grad_output, output, n, tau_input, false);
}

inline at::Tensor periodic_exponential_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& tau_input
) {
  return exponential_window_backward_impl(grad_output, output, n, tau_input, true);
}

// =============================================================================
// Hann-Poisson window
// =============================================================================

inline at::Tensor hann_poisson_window_impl(
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "hann_poisson_window: n must be non-negative, got ", n);
  TORCH_CHECK(alpha_input.dim() == 0, "hann_poisson_window: alpha must be a scalar tensor");
  TORCH_CHECK(!alpha_input.is_complex(), "hann_poisson_window: alpha must be real-valued");

  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(alpha_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "hann_poisson_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::hann_poisson<scalar_t>(i, n, alpha_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return hann_poisson_window_impl(n, alpha_input, false, dtype, layout, device);
}

inline at::Tensor periodic_hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return hann_poisson_window_impl(n, alpha_input, true, dtype, layout, device);
}

inline at::Tensor hann_poisson_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input,
  bool periodic
) {
  auto grad_alpha = at::zeros_like(alpha_input);

  if (n == 0) {
    return grad_alpha;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "hann_poisson_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t alpha_val = alpha_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      // Serial accumulation
      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::hann_poisson_backward<scalar_t>(
          grad_out_ptr[i], i, n, alpha_val, periodic, out_ptr[i]
        );
      }

      grad_alpha.fill_(accum);
    }
  );

  return grad_alpha;
}

inline at::Tensor hann_poisson_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return hann_poisson_window_backward_impl(grad_output, output, n, alpha_input, false);
}

inline at::Tensor periodic_hann_poisson_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return hann_poisson_window_backward_impl(grad_output, output, n, alpha_input, true);
}

// =============================================================================
// Generalized Normal window (two parameters: p and sigma)
// =============================================================================

inline at::Tensor generalized_normal_window_impl(
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
  TORCH_CHECK(!p_input.is_complex(), "generalized_normal_window: p must be real-valued");
  TORCH_CHECK(!sigma_input.is_complex(), "generalized_normal_window: sigma must be real-valued");

  // Determine output dtype from inputs
  auto promoted = at::result_type(p_input, sigma_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(p_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "generalized_normal_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t p_val = p_input.item<scalar_t>();
      scalar_t sigma_val = sigma_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::generalized_normal<scalar_t>(i, n, p_val, sigma_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_normal_window_impl(n, p_input, sigma_input, false, dtype, layout, device);
}

inline at::Tensor periodic_generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_normal_window_impl(n, p_input, sigma_input, true, dtype, layout, device);
}

inline std::tuple<at::Tensor, at::Tensor> generalized_normal_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  bool periodic
) {
  auto grad_p = at::zeros_like(p_input);
  auto grad_sigma = at::zeros_like(sigma_input);

  if (n == 0) {
    return std::make_tuple(grad_p, grad_sigma);
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "generalized_normal_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t p_val = p_input.item<scalar_t>();
      scalar_t sigma_val = sigma_input.item<scalar_t>();
      scalar_t accum_p = scalar_t(0);
      scalar_t accum_sigma = scalar_t(0);

      // Serial accumulation
      for (int64_t i = 0; i < n; ++i) {
        accum_p += kernel::window_function::generalized_normal_backward_p<scalar_t>(
          grad_out_ptr[i], i, n, p_val, sigma_val, periodic, out_ptr[i]
        );
        accum_sigma += kernel::window_function::generalized_normal_backward_sigma<scalar_t>(
          grad_out_ptr[i], i, n, p_val, sigma_val, periodic, out_ptr[i]
        );
      }

      grad_p.fill_(accum_p);
      grad_sigma.fill_(accum_sigma);
    }
  );

  return std::make_tuple(grad_p, grad_sigma);
}

inline std::tuple<at::Tensor, at::Tensor> generalized_normal_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input
) {
  return generalized_normal_window_backward_impl(grad_output, output, n, p_input, sigma_input, false);
}

inline std::tuple<at::Tensor, at::Tensor> periodic_generalized_normal_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input
) {
  return generalized_normal_window_backward_impl(grad_output, output, n, p_input, sigma_input, true);
}

// =============================================================================
// Kaiser window
// =============================================================================

inline at::Tensor kaiser_window_impl(
  int64_t n,
  const at::Tensor& beta_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "kaiser_window: n must be non-negative, got ", n);
  TORCH_CHECK(beta_input.dim() == 0, "kaiser_window: beta must be a scalar tensor");
  TORCH_CHECK(!beta_input.is_complex(), "kaiser_window: beta must be real-valued");

  auto out_dtype = dtype.value_or(beta_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(beta_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "kaiser_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t beta_val = beta_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::kaiser<scalar_t>(i, n, beta_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return kaiser_window_impl(n, beta_input, false, dtype, layout, device);
}

inline at::Tensor periodic_kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return kaiser_window_impl(n, beta_input, true, dtype, layout, device);
}

inline at::Tensor kaiser_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& beta_input,
  bool periodic
) {
  auto grad_beta = at::zeros_like(beta_input);

  if (n == 0) {
    return grad_beta;
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "kaiser_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t beta_val = beta_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      // Serial accumulation
      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::kaiser_backward<scalar_t>(
          grad_out_ptr[i], i, n, beta_val, periodic, out_ptr[i]
        );
      }

      grad_beta.fill_(accum);
    }
  );

  return grad_beta;
}

inline at::Tensor kaiser_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& beta_input
) {
  return kaiser_window_backward_impl(grad_output, output, n, beta_input, false);
}

inline at::Tensor periodic_kaiser_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& beta_input
) {
  return kaiser_window_backward_impl(grad_output, output, n, beta_input, true);
}

// =============================================================================
// Planck-taper window
// =============================================================================

inline at::Tensor planck_taper_window_impl(
  int64_t n,
  const at::Tensor& epsilon_input,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "planck_taper_window: n must be non-negative, got ", n);
  TORCH_CHECK(epsilon_input.dim() == 0, "planck_taper_window: epsilon must be a scalar tensor");
  TORCH_CHECK(!epsilon_input.is_complex(), "planck_taper_window: epsilon must be real-valued");

  auto out_dtype = dtype.value_or(epsilon_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(epsilon_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "planck_taper_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t epsilon_val = epsilon_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::planck_taper<scalar_t>(i, n, epsilon_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_taper_window_impl(n, epsilon_input, false, dtype, layout, device);
}

inline at::Tensor periodic_planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_taper_window_impl(n, epsilon_input, true, dtype, layout, device);
}

inline at::Tensor planck_taper_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  bool periodic
) {
  auto grad_epsilon = at::zeros_like(epsilon_input);

  if (n == 0) {
    return grad_epsilon;
  }

  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "planck_taper_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t epsilon_val = epsilon_input.item<scalar_t>();
      scalar_t accum = scalar_t(0);

      for (int64_t i = 0; i < n; ++i) {
        accum += kernel::window_function::planck_taper_backward<scalar_t>(
          grad_out_ptr[i], i, n, epsilon_val, periodic, out_ptr[i]
        );
      }

      grad_epsilon.fill_(accum);
    }
  );

  return grad_epsilon;
}

inline at::Tensor planck_taper_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input
) {
  return planck_taper_window_backward_impl(grad_output, output, n, epsilon_input, false);
}

inline at::Tensor periodic_planck_taper_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input
) {
  return planck_taper_window_backward_impl(grad_output, output, n, epsilon_input, true);
}

// =============================================================================
// Planck-Bessel window (two parameters: epsilon and beta)
// =============================================================================

inline at::Tensor planck_bessel_window_impl(
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
  TORCH_CHECK(!epsilon_input.is_complex(), "planck_bessel_window: epsilon must be real-valued");
  TORCH_CHECK(!beta_input.is_complex(), "planck_bessel_window: beta must be real-valued");

  // Determine output dtype from inputs
  auto promoted = at::result_type(epsilon_input, beta_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(epsilon_input.device()));

  auto output = at::empty({n}, options);

  if (n == 0) {
    return output;
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    output.scalar_type(),
    "planck_bessel_window",
    [&] {
      auto* out_ptr = output.data_ptr<scalar_t>();
      scalar_t epsilon_val = epsilon_input.item<scalar_t>();
      scalar_t beta_val = beta_input.item<scalar_t>();
      at::parallel_for(0, n, 1024, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          out_ptr[i] = kernel::window_function::planck_bessel<scalar_t>(i, n, epsilon_val, beta_val, periodic);
        }
      });
    }
  );

  return output;
}

inline at::Tensor planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_bessel_window_impl(n, epsilon_input, beta_input, false, dtype, layout, device);
}

inline at::Tensor periodic_planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_bessel_window_impl(n, epsilon_input, beta_input, true, dtype, layout, device);
}

inline std::tuple<at::Tensor, at::Tensor> planck_bessel_window_backward_impl(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  bool periodic
) {
  auto grad_epsilon = at::zeros_like(epsilon_input);
  auto grad_beta = at::zeros_like(beta_input);

  if (n == 0) {
    return std::make_tuple(grad_epsilon, grad_beta);
  }

  // Ensure contiguous tensors for data_ptr access
  at::Tensor grad_output_contig = grad_output.contiguous();
  at::Tensor output_contig = output.contiguous();

  AT_DISPATCH_FLOATING_TYPES_AND2(
    at::kBFloat16, at::kHalf,
    grad_output_contig.scalar_type(),
    "planck_bessel_window_backward",
    [&] {
      auto* grad_out_ptr = grad_output_contig.data_ptr<scalar_t>();
      auto* out_ptr = output_contig.data_ptr<scalar_t>();
      scalar_t epsilon_val = epsilon_input.item<scalar_t>();
      scalar_t beta_val = beta_input.item<scalar_t>();
      scalar_t accum_epsilon = scalar_t(0);
      scalar_t accum_beta = scalar_t(0);

      // Serial accumulation
      for (int64_t i = 0; i < n; ++i) {
        scalar_t g_eps, g_beta;
        kernel::window_function::planck_bessel_backward<scalar_t>(
          grad_out_ptr[i], i, n, epsilon_val, beta_val, periodic, out_ptr[i],
          g_eps, g_beta
        );
        accum_epsilon += g_eps;
        accum_beta += g_beta;
      }

      grad_epsilon.fill_(accum_epsilon);
      grad_beta.fill_(accum_beta);
    }
  );

  return std::make_tuple(grad_epsilon, grad_beta);
}

inline std::tuple<at::Tensor, at::Tensor> planck_bessel_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input
) {
  return planck_bessel_window_backward_impl(grad_output, output, n, epsilon_input, beta_input, false);
}

inline std::tuple<at::Tensor, at::Tensor> periodic_planck_bessel_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input
) {
  return planck_bessel_window_backward_impl(grad_output, output, n, epsilon_input, beta_input, true);
}

}  // namespace torchscience::cpu::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CPU, m) {
  m.impl("hann_window", torchscience::cpu::window_function::hann_window);
  m.impl("periodic_hann_window", torchscience::cpu::window_function::periodic_hann_window);
  m.impl("hamming_window", torchscience::cpu::window_function::hamming_window);
  m.impl("periodic_hamming_window", torchscience::cpu::window_function::periodic_hamming_window);
  m.impl("blackman_window", torchscience::cpu::window_function::blackman_window);
  m.impl("periodic_blackman_window", torchscience::cpu::window_function::periodic_blackman_window);
  m.impl("bartlett_window", torchscience::cpu::window_function::bartlett_window);
  m.impl("periodic_bartlett_window", torchscience::cpu::window_function::periodic_bartlett_window);
  m.impl("cosine_window", torchscience::cpu::window_function::cosine_window);
  m.impl("periodic_cosine_window", torchscience::cpu::window_function::periodic_cosine_window);
  m.impl("nuttall_window", torchscience::cpu::window_function::nuttall_window);
  m.impl("periodic_nuttall_window", torchscience::cpu::window_function::periodic_nuttall_window);
  m.impl("triangular_window", torchscience::cpu::window_function::triangular_window);
  m.impl("periodic_triangular_window", torchscience::cpu::window_function::periodic_triangular_window);
  m.impl("welch_window", torchscience::cpu::window_function::welch_window);
  m.impl("periodic_welch_window", torchscience::cpu::window_function::periodic_welch_window);
  m.impl("parzen_window", torchscience::cpu::window_function::parzen_window);
  m.impl("periodic_parzen_window", torchscience::cpu::window_function::periodic_parzen_window);
  m.impl("blackman_harris_window", torchscience::cpu::window_function::blackman_harris_window);
  m.impl("periodic_blackman_harris_window", torchscience::cpu::window_function::periodic_blackman_harris_window);
  m.impl("flat_top_window", torchscience::cpu::window_function::flat_top_window);
  m.impl("periodic_flat_top_window", torchscience::cpu::window_function::periodic_flat_top_window);
  m.impl("sine_window", torchscience::cpu::window_function::sine_window);
  m.impl("periodic_sine_window", torchscience::cpu::window_function::periodic_sine_window);
  m.impl("bartlett_hann_window", torchscience::cpu::window_function::bartlett_hann_window);
  m.impl("periodic_bartlett_hann_window", torchscience::cpu::window_function::periodic_bartlett_hann_window);
  m.impl("lanczos_window", torchscience::cpu::window_function::lanczos_window);
  m.impl("periodic_lanczos_window", torchscience::cpu::window_function::periodic_lanczos_window);

  m.impl("gaussian_window", torchscience::cpu::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::cpu::window_function::periodic_gaussian_window);
  m.impl("gaussian_window_backward", torchscience::cpu::window_function::gaussian_window_backward);
  m.impl("periodic_gaussian_window_backward", torchscience::cpu::window_function::periodic_gaussian_window_backward);

  m.impl("general_hamming_window", torchscience::cpu::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::cpu::window_function::periodic_general_hamming_window);
  m.impl("general_hamming_window_backward", torchscience::cpu::window_function::general_hamming_window_backward);
  m.impl("periodic_general_hamming_window_backward", torchscience::cpu::window_function::periodic_general_hamming_window_backward);

  m.impl("general_cosine_window", torchscience::cpu::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::cpu::window_function::periodic_general_cosine_window);
  m.impl("general_cosine_window_backward", torchscience::cpu::window_function::general_cosine_window_backward);
  m.impl("periodic_general_cosine_window_backward", torchscience::cpu::window_function::periodic_general_cosine_window_backward);

  m.impl("tukey_window", torchscience::cpu::window_function::tukey_window);
  m.impl("periodic_tukey_window", torchscience::cpu::window_function::periodic_tukey_window);
  m.impl("tukey_window_backward", torchscience::cpu::window_function::tukey_window_backward);
  m.impl("periodic_tukey_window_backward", torchscience::cpu::window_function::periodic_tukey_window_backward);

  m.impl("exponential_window", torchscience::cpu::window_function::exponential_window);
  m.impl("periodic_exponential_window", torchscience::cpu::window_function::periodic_exponential_window);
  m.impl("exponential_window_backward", torchscience::cpu::window_function::exponential_window_backward);
  m.impl("periodic_exponential_window_backward", torchscience::cpu::window_function::periodic_exponential_window_backward);

  m.impl("hann_poisson_window", torchscience::cpu::window_function::hann_poisson_window);
  m.impl("periodic_hann_poisson_window", torchscience::cpu::window_function::periodic_hann_poisson_window);
  m.impl("hann_poisson_window_backward", torchscience::cpu::window_function::hann_poisson_window_backward);
  m.impl("periodic_hann_poisson_window_backward", torchscience::cpu::window_function::periodic_hann_poisson_window_backward);

  m.impl("generalized_normal_window", torchscience::cpu::window_function::generalized_normal_window);
  m.impl("periodic_generalized_normal_window", torchscience::cpu::window_function::periodic_generalized_normal_window);
  m.impl("generalized_normal_window_backward", torchscience::cpu::window_function::generalized_normal_window_backward);
  m.impl("periodic_generalized_normal_window_backward", torchscience::cpu::window_function::periodic_generalized_normal_window_backward);

  m.impl("kaiser_window", torchscience::cpu::window_function::kaiser_window);
  m.impl("periodic_kaiser_window", torchscience::cpu::window_function::periodic_kaiser_window);
  m.impl("kaiser_window_backward", torchscience::cpu::window_function::kaiser_window_backward);
  m.impl("periodic_kaiser_window_backward", torchscience::cpu::window_function::periodic_kaiser_window_backward);

  m.impl("planck_taper_window", torchscience::cpu::window_function::planck_taper_window);
  m.impl("periodic_planck_taper_window", torchscience::cpu::window_function::periodic_planck_taper_window);
  m.impl("planck_taper_window_backward", torchscience::cpu::window_function::planck_taper_window_backward);
  m.impl("periodic_planck_taper_window_backward", torchscience::cpu::window_function::periodic_planck_taper_window_backward);

  m.impl("planck_bessel_window", torchscience::cpu::window_function::planck_bessel_window);
  m.impl("periodic_planck_bessel_window", torchscience::cpu::window_function::periodic_planck_bessel_window);
  m.impl("planck_bessel_window_backward", torchscience::cpu::window_function::planck_bessel_window_backward);
  m.impl("periodic_planck_bessel_window_backward", torchscience::cpu::window_function::periodic_planck_bessel_window_backward);
}
