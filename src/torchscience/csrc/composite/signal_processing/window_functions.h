#pragma once

#include <vector>
#include <torch/extension.h>
#include "../../cpu/creation_operators.h"
#include "../../meta/creation_operators.h"

namespace torchscience::composite::window_function {

namespace {

struct RectangularWindowTraits {
    static std::vector<int64_t> output_shape(int64_t n) {
        return {n};
    }

    template<typename scalar_t>
    static void kernel(scalar_t* output, int64_t numel, int64_t n) {
        (void)n;  // n == numel for rectangular window
        for (int64_t i = 0; i < numel; ++i) {
            output[i] = scalar_t(1);
        }
    }
};

}  // anonymous namespace

// Composite implementation that routes to appropriate backend based on device
inline at::Tensor rectangular_window(
  int64_t n,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad
) {
  at::Device target_device = device.value_or(at::kCPU);

  if (target_device.type() == at::kMeta) {
    return torchscience::meta::MetaCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  } else {
    return torchscience::cpu::CPUCreationOperator<RectangularWindowTraits>::forward<int64_t>(
      n, dtype, layout, device, requires_grad
    );
  }
}

// =============================================================================
// Routing macros for parameterless windows
// =============================================================================

#define DEFINE_PARAMETERLESS_WINDOW_ROUTER(name)                                \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  at::Device target_device = device.value_or(at::kCPU);                         \
  c10::DispatchKeySet ks;                                                       \
  if (target_device.type() == at::kMeta) {                                      \
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);                           \
  } else if (target_device.type() == at::kCUDA) {                               \
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);                           \
  } else {                                                                      \
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);                            \
  }                                                                             \
  return c10::Dispatcher::singleton()                                           \
    .findSchemaOrThrow("torchscience::" #name "_window", "")                    \
    .typed<at::Tensor(int64_t, c10::optional<at::ScalarType>,                   \
                      c10::optional<at::Layout>, c10::optional<at::Device>,     \
                      bool)>()                                                  \
    .redispatch(ks, n, dtype, layout, device, requires_grad);                   \
}                                                                               \
                                                                                \
inline at::Tensor periodic_##name##_window(                                     \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  at::Device target_device = device.value_or(at::kCPU);                         \
  c10::DispatchKeySet ks;                                                       \
  if (target_device.type() == at::kMeta) {                                      \
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);                           \
  } else if (target_device.type() == at::kCUDA) {                               \
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);                           \
  } else {                                                                      \
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);                            \
  }                                                                             \
  return c10::Dispatcher::singleton()                                           \
    .findSchemaOrThrow("torchscience::periodic_" #name "_window", "")           \
    .typed<at::Tensor(int64_t, c10::optional<at::ScalarType>,                   \
                      c10::optional<at::Layout>, c10::optional<at::Device>,     \
                      bool)>()                                                  \
    .redispatch(ks, n, dtype, layout, device, requires_grad);                   \
}

DEFINE_PARAMETERLESS_WINDOW_ROUTER(hann)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(hamming)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(blackman)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(bartlett)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(cosine)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(nuttall)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(triangular)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(welch)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(parzen)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(blackman_harris)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(flat_top)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(sine)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(bartlett_hann)
DEFINE_PARAMETERLESS_WINDOW_ROUTER(lanczos)

#undef DEFINE_PARAMETERLESS_WINDOW_ROUTER

// =============================================================================
// Routing for parameterized windows
// =============================================================================

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  // For parameterized windows, device from tensor or explicit device param
  at::Device target_device = device.value_or(std_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::gaussian_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, std_input, dtype, layout, device);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(std_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_gaussian_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, std_input, dtype, layout, device);
}

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::general_hamming_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_general_hamming_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(coeffs_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::general_cosine_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, coeffs_input, dtype, layout, device);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(coeffs_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_general_cosine_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, coeffs_input, dtype, layout, device);
}

inline at::Tensor tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::tukey_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor periodic_tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_tukey_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(tau_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::exponential_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, tau_input, dtype, layout, device);
}

inline at::Tensor periodic_exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(tau_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_exponential_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, tau_input, dtype, layout, device);
}

inline at::Tensor hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::hann_poisson_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor periodic_hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(alpha_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_hann_poisson_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, alpha_input, dtype, layout, device);
}

inline at::Tensor generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(p_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::generalized_normal_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, p_input, sigma_input, dtype, layout, device);
}

inline at::Tensor periodic_generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(p_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_generalized_normal_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, p_input, sigma_input, dtype, layout, device);
}

inline at::Tensor kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(beta_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::kaiser_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, beta_input, dtype, layout, device);
}

inline at::Tensor periodic_kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(beta_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_kaiser_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, beta_input, dtype, layout, device);
}

inline at::Tensor planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(epsilon_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::planck_taper_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, epsilon_input, dtype, layout, device);
}

inline at::Tensor periodic_planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(epsilon_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_planck_taper_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, epsilon_input, dtype, layout, device);
}

inline at::Tensor planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(epsilon_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::planck_bessel_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, epsilon_input, beta_input, dtype, layout, device);
}

inline at::Tensor periodic_planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  at::Device target_device = device.value_or(epsilon_input.device());
  c10::DispatchKeySet ks;
  if (target_device.type() == at::kMeta) {
    ks = c10::DispatchKeySet(c10::DispatchKey::Meta);
  } else if (target_device.type() == at::kCUDA) {
    ks = c10::DispatchKeySet(c10::DispatchKey::CUDA);
  } else {
    ks = c10::DispatchKeySet(c10::DispatchKey::CPU);
  }
  return c10::Dispatcher::singleton()
    .findSchemaOrThrow("torchscience::periodic_planck_bessel_window", "")
    .typed<at::Tensor(int64_t, const at::Tensor&, const at::Tensor&, c10::optional<at::ScalarType>,
                      c10::optional<at::Layout>, c10::optional<at::Device>)>()
    .redispatch(ks, n, epsilon_input, beta_input, dtype, layout, device);
}

// =============================================================================
// FFT-based windows (Pattern C) - use CompositeImplicitAutograd
// =============================================================================

namespace {

// Helper: Evaluate Chebyshev polynomial T_n(x) analytically
// Uses at::where for efficiency
// For |x| <= 1: T_n(x) = cos(n * arccos(x))
// For x > 1:    T_n(x) = cosh(n * arccosh(x))
// For x < -1:   T_n(x) = (-1)^n * cosh(n * arccosh(-x))
inline at::Tensor chebyshev_t(double order, const at::Tensor& x) {
  double sign = std::cos(M_PI * order);  // (-1)^n for integer n

  auto abs_x = at::abs(x);

  // Clamp for numerical stability at boundaries
  auto x_clamped = at::clamp(x, -1.0, 1.0);

  return at::where(abs_x <= 1,
      at::cos(order * at::acos(x_clamped)),
      at::where(x > 1,
          at::cosh(order * at::acosh(x)),
          sign * at::cosh(order * at::acosh(-x))));
}

// Helper: Evaluate Gegenbauer polynomial C_n^{mu}(x) using recurrence relation
// C_0^{mu}(x) = 1
// C_1^{mu}(x) = 2*mu*x
// C_{k+1}^{mu}(x) = (2*(k+mu)/(k+1)) * x * C_k^{mu}(x) - ((k+2*mu-1)/(k+1)) * C_{k-1}^{mu}(x)
inline at::Tensor gegenbauer_c(int64_t order, const at::Tensor& mu, const at::Tensor& x) {
  if (order == 0) {
    return at::ones_like(x);
  }
  if (order == 1) {
    return 2.0 * mu * x;
  }

  auto c_prev = at::ones_like(x);   // C_0
  auto c_curr = 2.0 * mu * x;       // C_1

  for (int64_t k = 1; k < order; ++k) {
    // Compute C_{k+1} from C_k and C_{k-1}
    double k_f = static_cast<double>(k);
    auto a_k = 2.0 * (k_f + mu) / (k_f + 1.0);
    auto b_k = (k_f + 2.0 * mu - 1.0) / (k_f + 1.0);
    auto c_next = a_k * x * c_curr - b_k * c_prev;
    c_prev = c_curr;
    c_curr = c_next;
  }

  return c_curr;
}

}  // anonymous namespace

inline at::Tensor dolph_chebyshev_window_impl(
  int64_t n,
  const at::Tensor& attenuation,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "dolph_chebyshev_window: n must be non-negative, got ", n);
  TORCH_CHECK(attenuation.dim() == 0, "dolph_chebyshev_window: attenuation must be a scalar tensor");
  TORCH_CHECK(!attenuation.is_complex(), "dolph_chebyshev_window: attenuation must be real-valued");
  // Note: attenuation > 0 check is done via Python wrapper to allow proper ValueError

  auto target_dtype = dtype.value_or(attenuation.scalar_type());
  auto target_device = device.value_or(attenuation.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Ensure we work in float64 for numerical precision in intermediate calculations
  auto work_options = at::TensorOptions()
    .dtype(at::ScalarType::Double)
    .device(target_device);
  auto attenuation_f64 = attenuation.to(at::ScalarType::Double);

  // Compute beta from attenuation (dB)
  // beta = cosh(acosh(10^(|A|/20)) / (N-1))
  double order = static_cast<double>(n - 1);
  auto ten = at::scalar_tensor(10.0, work_options);
  auto amp_ratio = at::pow(ten, at::abs(attenuation_f64) / 20.0);
  auto beta = at::cosh(at::acosh(amp_ratio) / order);

  // Sample points for Chebyshev polynomial
  auto k = at::arange(n, work_options);
  auto x = beta * at::cos(M_PI * k / n);

  // Evaluate T_{N-1}(x)
  auto p = chebyshev_t(order, x);

  // Compute window via FFT
  at::Tensor window;
  if (n % 2 == 1) {
    // Odd N: direct FFT
    auto w = at::real(at::fft_fft(p));
    auto half = (n + 1) / 2;
    auto w_half = w.slice(0, 0, half);
    window = at::cat({w_half.slice(0, 1, half).flip(0), w_half});
  } else {
    // Even N: apply phase shift before FFT
    // Need complex tensor for phase
    auto complex_options = at::TensorOptions()
      .dtype(at::ScalarType::ComplexDouble)
      .device(target_device);
    auto imag_unit = at::tensor(c10::complex<double>(0.0, 1.0), complex_options);
    auto phase = at::exp(imag_unit * (M_PI / n) * k);
    auto p_complex = p.to(at::ScalarType::ComplexDouble);
    auto p_shifted = p_complex * phase;
    auto w = at::real(at::fft_fft(p_shifted));
    auto half = n / 2 + 1;
    window = at::cat({w.slice(0, 1, half).flip(0), w.slice(0, 1, half)});
  }

  // Normalize so maximum is 1
  window = window / window.abs().max();

  // Convert to target dtype
  if (target_dtype != at::ScalarType::Double) {
    window = window.to(target_dtype);
  }

  return window;
}

inline at::Tensor dolph_chebyshev_window(
  int64_t n,
  const at::Tensor& attenuation,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return dolph_chebyshev_window_impl(n, attenuation, false, dtype, layout, device);
}

inline at::Tensor periodic_dolph_chebyshev_window(
  int64_t n,
  const at::Tensor& attenuation,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  // Periodic window: compute symmetric window of length n+1, drop last sample
  if (n == 0) {
    auto target_dtype = dtype.value_or(attenuation.scalar_type());
    auto options = at::TensorOptions()
      .dtype(target_dtype)
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(attenuation.device()));
    return at::empty({0}, options);
  }

  if (n == 1) {
    auto target_dtype = dtype.value_or(attenuation.scalar_type());
    auto options = at::TensorOptions()
      .dtype(target_dtype)
      .layout(layout.value_or(at::kStrided))
      .device(device.value_or(attenuation.device()));
    return at::ones({1}, options);
  }

  auto window_extended = dolph_chebyshev_window_impl(n + 1, attenuation, false, dtype, layout, device);
  return window_extended.slice(0, 0, n);
}

// =============================================================================
// Ultraspherical (Gegenbauer) window - Pattern C
// Uses Fourier series summation with Gegenbauer polynomials
// =============================================================================

inline at::Tensor ultraspherical_window_impl(
  int64_t n,
  const at::Tensor& mu,
  const at::Tensor& x_mu,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "ultraspherical_window: n must be non-negative, got ", n);
  TORCH_CHECK(mu.dim() == 0, "ultraspherical_window: mu must be a scalar tensor");
  TORCH_CHECK(x_mu.dim() == 0, "ultraspherical_window: x_mu must be a scalar tensor");
  TORCH_CHECK(!mu.is_complex(), "ultraspherical_window: mu must be real-valued");
  TORCH_CHECK(!x_mu.is_complex(), "ultraspherical_window: x_mu must be real-valued");
  // Note: mu > 0 and x_mu > 1 checks are done via Python wrapper for proper ValueError

  auto target_dtype = dtype.value_or(mu.scalar_type());
  auto target_device = device.value_or(mu.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Ensure we work in the target dtype for all computations
  auto work_options = at::TensorOptions()
    .dtype(target_dtype)
    .device(target_device);

  // Convert parameters to target dtype
  auto mu_t = mu.to(target_dtype);
  auto x_mu_t = x_mu.to(target_dtype);

  // Sample indices for window output
  auto k = at::arange(n, work_options);

  // Center and divisor differ for symmetric vs periodic
  // Symmetric: center = (n-1)/2, divisor = n-1
  // Periodic: center = n/2, divisor = n
  double center = periodic ? n / 2.0 : (n - 1) / 2.0;
  double divisor = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);

  // Initialize window to zeros
  auto window = at::zeros({n}, work_options);

  // Number of frequency components to sum
  int64_t n_freqs = n / 2 + 1;

  // Gegenbauer polynomial order
  int64_t order = n - 1;

  // Precompute C_{N-1}^{mu}(x_mu) - the normalization factor
  auto c_n_x_mu = gegenbauer_c(order, mu_t, x_mu_t.unsqueeze(0)).squeeze(0);

  for (int64_t m = 0; m < n_freqs; ++m) {
    // Frequency (normalized angular frequency)
    double omega = (divisor > 0) ? (M_PI * m / divisor) : 0.0;

    // Argument to Gegenbauer polynomial: x_mu * cos(omega)
    auto omega_t = at::scalar_tensor(std::cos(omega), work_options);
    auto arg = x_mu_t * omega_t;

    // Evaluate frequency response: C_{N-1}^{mu}(arg) / C_{N-1}^{mu}(x_mu)
    auto c_n_arg = gegenbauer_c(order, mu_t, arg.unsqueeze(0)).squeeze(0);
    auto freq_mag = c_n_arg / c_n_x_mu;

    // Compute cosine term for each output sample
    // cos(omega * (k - center))
    auto cosine_term = at::cos(omega * (k - center));

    // Weighting: DC and Nyquist (if present) are weighted by 1, others by 2
    double weight = (m == 0 || (n % 2 == 0 && m == n / 2)) ? 1.0 : 2.0;

    window = window + weight * freq_mag * cosine_term;
  }

  // Normalize so maximum is 1
  window = window / window.abs().max();

  return window;
}

inline at::Tensor ultraspherical_window(
  int64_t n,
  const at::Tensor& mu,
  const at::Tensor& x_mu,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return ultraspherical_window_impl(n, mu, x_mu, false, dtype, layout, device);
}

inline at::Tensor periodic_ultraspherical_window(
  int64_t n,
  const at::Tensor& mu,
  const at::Tensor& x_mu,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return ultraspherical_window_impl(n, mu, x_mu, true, dtype, layout, device);
}

// =============================================================================
// Phase 5: Eigenvalue-based and polynomial windows (CompositeImplicitAutograd)
// =============================================================================

// Helper function for DPSS: compute principal eigenvector of tridiagonal matrix
inline at::Tensor dpss_tridiagonal_eigenvector(
  int64_t n,
  const at::Tensor& nw,
  at::ScalarType dtype,
  at::Device device
) {
  // Compute W from NW
  auto w = nw / static_cast<double>(n);

  auto options = at::TensorOptions().dtype(dtype).device(device);

  // Build index tensor
  auto i = at::arange(n, options);

  // Diagonal elements: ((N-1)/2 - i)^2 * cos(2*pi*W)
  double center = (n - 1) / 2.0;
  auto diagonal = at::pow(center - i, 2) * at::cos(2.0 * M_PI * w);

  // Off-diagonal elements: j*(N-j)/2 for j = 1, ..., N-1
  auto j = at::arange(1, n, options);
  auto off_diagonal = j * (static_cast<double>(n) - j) / 2.0;

  // Construct the full tridiagonal matrix
  auto matrix = at::diag(diagonal);
  matrix = matrix + at::diag(off_diagonal, 1);
  matrix = matrix + at::diag(off_diagonal, -1);

  // Compute eigenvalues and eigenvectors
  auto result = at::linalg_eigh(matrix);
  auto eigenvectors = std::get<1>(result);

  // eigh returns eigenvalues in ascending order, we want the largest
  // The principal DPSS (zeroth order) corresponds to the largest eigenvalue
  auto principal_eigenvector = eigenvectors.select(1, n - 1);

  // Ensure the window is positive at the center (conventional normalization)
  int64_t center_idx = n / 2;
  auto center_val = principal_eigenvector.index({center_idx});
  auto sign_mask = center_val < 0;
  principal_eigenvector = at::where(
    sign_mask.expand_as(principal_eigenvector),
    -principal_eigenvector,
    principal_eigenvector
  );

  return principal_eigenvector;
}

inline at::Tensor discrete_prolate_spheroidal_sequence_window_impl(
  int64_t n,
  const at::Tensor& nw,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "discrete_prolate_spheroidal_sequence_window: n must be non-negative, got ", n);
  TORCH_CHECK(nw.dim() == 0, "discrete_prolate_spheroidal_sequence_window: nw must be a scalar tensor");
  TORCH_CHECK(!nw.is_complex(), "discrete_prolate_spheroidal_sequence_window: nw must be real-valued");
  // Note: nw > 0 check is done via Python wrapper for proper ValueError

  auto target_dtype = dtype.value_or(nw.scalar_type());
  auto target_device = device.value_or(nw.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Convert nw to target dtype
  auto nw_t = nw.to(target_dtype);

  // Compute the principal DPSS eigenvector
  // For periodic, use n+1 and drop last point
  int64_t work_n = periodic ? n + 1 : n;
  auto window = dpss_tridiagonal_eigenvector(work_n, nw_t, target_dtype, target_device);

  if (periodic) {
    window = window.slice(0, 0, n);
  }

  // Normalize so maximum value is 1
  window = window / window.abs().max();

  return window;
}

inline at::Tensor discrete_prolate_spheroidal_sequence_window(
  int64_t n,
  const at::Tensor& nw,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return discrete_prolate_spheroidal_sequence_window_impl(n, nw, false, dtype, layout, device);
}

inline at::Tensor periodic_discrete_prolate_spheroidal_sequence_window(
  int64_t n,
  const at::Tensor& nw,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return discrete_prolate_spheroidal_sequence_window_impl(n, nw, true, dtype, layout, device);
}

// =============================================================================
// Approximate Confined Gaussian Window
// =============================================================================

inline at::Tensor approximate_confined_gaussian_window_impl(
  int64_t n,
  const at::Tensor& sigma,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "approximate_confined_gaussian_window: n must be non-negative, got ", n);
  TORCH_CHECK(sigma.dim() == 0, "approximate_confined_gaussian_window: sigma must be a scalar tensor");
  TORCH_CHECK(!sigma.is_complex(), "approximate_confined_gaussian_window: sigma must be real-valued");

  auto target_dtype = dtype.value_or(sigma.scalar_type());
  auto target_device = device.value_or(sigma.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Convert sigma to target dtype
  auto sigma_t = sigma.to(target_dtype);

  auto k = at::arange(n, options);

  if (periodic) {
    // Periodic version uses different formula:
    // t = (k - n/2) / n (normalized to [-0.5, 0.5))
    // g(t) = exp(-0.5 * (t / sigma)^2)
    // window = g(t) - g(-0.5)
    double denom = static_cast<double>(n);
    double center = n / 2.0;

    auto t = (k - center) / denom;
    auto g_t = at::exp(-0.5 * at::pow(t / sigma_t, 2));

    // g(-0.5) = exp(-0.5 * (-0.5/sigma)^2) = exp(-0.125/sigma^2)
    auto g_boundary = at::exp(-0.5 * at::pow(at::scalar_tensor(-0.5, options) / sigma_t, 2));

    auto window = g_t - g_boundary;

    // Clamp to ensure non-negative
    return at::clamp(window, 0.0);
  } else {
    // Symmetric version:
    // center = (n-1)/2
    // G[k] = exp(-0.5 * ((k - center) / (sigma * center))^2)
    // G_endpoint = exp(-0.5 / sigma^2)
    // window = (G - G_endpoint) / (1 - G_endpoint)
    double center = (n - 1) / 2.0;

    auto normalized_x = (k - center) / (sigma_t * center);
    auto G = at::exp(-0.5 * normalized_x * normalized_x);

    auto G_endpoint = at::exp(-0.5 / (sigma_t * sigma_t));
    auto window = (G - G_endpoint) / (1.0 - G_endpoint);

    return window;
  }
}

inline at::Tensor approximate_confined_gaussian_window(
  int64_t n,
  const at::Tensor& sigma,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return approximate_confined_gaussian_window_impl(n, sigma, false, dtype, layout, device);
}

inline at::Tensor periodic_approximate_confined_gaussian_window(
  int64_t n,
  const at::Tensor& sigma,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return approximate_confined_gaussian_window_impl(n, sigma, true, dtype, layout, device);
}

// =============================================================================
// Confined Gaussian Window
// =============================================================================

inline at::Tensor confined_gaussian_window_impl(
  int64_t n,
  const at::Tensor& sigma,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "confined_gaussian_window: n must be non-negative, got ", n);
  TORCH_CHECK(sigma.dim() == 0, "confined_gaussian_window: sigma must be a scalar tensor");
  TORCH_CHECK(!sigma.is_complex(), "confined_gaussian_window: sigma must be real-valued");

  auto target_dtype = dtype.value_or(sigma.scalar_type());
  auto target_device = device.value_or(sigma.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Convert sigma to target dtype
  auto sigma_t = sigma.to(target_dtype);

  // For symmetric: denominator = n-1
  // For periodic: denominator = n
  double denom = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);
  double center = periodic ? n / 2.0 : (n - 1) / 2.0;

  // Compute normalized positions t in [-0.5, 0.5]
  auto k = at::arange(n, options);
  auto t = (k - center) / denom;

  // Gaussian function helper: G(x) = exp(-0.5 * (x / sigma)^2)
  auto gaussian = [&sigma_t](const at::Tensor& x) {
    return at::exp(-0.5 * at::pow(x / sigma_t, 2));
  };

  // Compute the confined Gaussian window
  // w[k] = G(t_k) - G(0.5) * (G(t_k - 1) + G(t_k + 1)) / (G(0.5) + G(1.5))
  auto g_t = gaussian(t);
  auto g_t_minus_1 = gaussian(t - 1.0);
  auto g_t_plus_1 = gaussian(t + 1.0);

  auto half = at::scalar_tensor(0.5, options);
  auto three_half = at::scalar_tensor(1.5, options);
  auto g_half = gaussian(half);
  auto g_three_half = gaussian(three_half);

  auto correction = g_half / (g_half + g_three_half);
  auto window = g_t - correction * (g_t_minus_1 + g_t_plus_1);

  return window;
}

inline at::Tensor confined_gaussian_window(
  int64_t n,
  const at::Tensor& sigma,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return confined_gaussian_window_impl(n, sigma, false, dtype, layout, device);
}

inline at::Tensor periodic_confined_gaussian_window(
  int64_t n,
  const at::Tensor& sigma,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return confined_gaussian_window_impl(n, sigma, true, dtype, layout, device);
}

// =============================================================================
// Generalized Adaptive Polynomial Window
// =============================================================================

inline at::Tensor generalized_adaptive_polynomial_window_impl(
  int64_t n,
  const at::Tensor& alpha,
  const at::Tensor& beta,
  bool periodic,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  TORCH_CHECK(n >= 0, "generalized_adaptive_polynomial_window: n must be non-negative, got ", n);
  TORCH_CHECK(alpha.dim() == 0, "generalized_adaptive_polynomial_window: alpha must be a scalar tensor");
  TORCH_CHECK(beta.dim() == 0, "generalized_adaptive_polynomial_window: beta must be a scalar tensor");
  TORCH_CHECK(!alpha.is_complex(), "generalized_adaptive_polynomial_window: alpha must be real-valued");
  TORCH_CHECK(!beta.is_complex(), "generalized_adaptive_polynomial_window: beta must be real-valued");

  auto target_dtype = dtype.value_or(alpha.scalar_type());
  auto target_device = device.value_or(alpha.device());
  auto options = at::TensorOptions()
    .dtype(target_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(target_device);

  if (n == 0) {
    return at::empty({0}, options);
  }

  if (n == 1) {
    return at::ones({1}, options);
  }

  // Convert parameters to target dtype
  auto alpha_t = alpha.to(target_dtype);
  auto beta_t = beta.to(target_dtype);

  // For symmetric: denominator = n-1
  // For periodic: denominator = n
  double denom = periodic ? static_cast<double>(n) : static_cast<double>(n - 1);

  auto k = at::arange(n, options);

  // Normalized position x in [-1, 1]
  auto x = 2.0 * k / denom - 1.0;

  // w[k] = (1 - |x|^alpha)^beta
  // Use clamp to ensure numerical stability
  auto abs_x_alpha = at::pow(at::abs(x), alpha_t);
  auto window = at::pow(at::clamp(1.0 - abs_x_alpha, 0.0), beta_t);

  return window;
}

inline at::Tensor generalized_adaptive_polynomial_window(
  int64_t n,
  const at::Tensor& alpha,
  const at::Tensor& beta,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_adaptive_polynomial_window_impl(n, alpha, beta, false, dtype, layout, device);
}

inline at::Tensor periodic_generalized_adaptive_polynomial_window(
  int64_t n,
  const at::Tensor& alpha,
  const at::Tensor& beta,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_adaptive_polynomial_window_impl(n, alpha, beta, true, dtype, layout, device);
}

} // namespace torchscience::composite::window_function

// =============================================================================
// CompositeExplicitAutograd registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl("rectangular_window", &torchscience::composite::window_function::rectangular_window);

  module.impl("hann_window", &torchscience::composite::window_function::hann_window);
  module.impl("periodic_hann_window", &torchscience::composite::window_function::periodic_hann_window);
  module.impl("hamming_window", &torchscience::composite::window_function::hamming_window);
  module.impl("periodic_hamming_window", &torchscience::composite::window_function::periodic_hamming_window);
  module.impl("blackman_window", &torchscience::composite::window_function::blackman_window);
  module.impl("periodic_blackman_window", &torchscience::composite::window_function::periodic_blackman_window);
  module.impl("bartlett_window", &torchscience::composite::window_function::bartlett_window);
  module.impl("periodic_bartlett_window", &torchscience::composite::window_function::periodic_bartlett_window);
  module.impl("cosine_window", &torchscience::composite::window_function::cosine_window);
  module.impl("periodic_cosine_window", &torchscience::composite::window_function::periodic_cosine_window);
  module.impl("nuttall_window", &torchscience::composite::window_function::nuttall_window);
  module.impl("periodic_nuttall_window", &torchscience::composite::window_function::periodic_nuttall_window);
  module.impl("triangular_window", &torchscience::composite::window_function::triangular_window);
  module.impl("periodic_triangular_window", &torchscience::composite::window_function::periodic_triangular_window);
  module.impl("welch_window", &torchscience::composite::window_function::welch_window);
  module.impl("periodic_welch_window", &torchscience::composite::window_function::periodic_welch_window);
  module.impl("parzen_window", &torchscience::composite::window_function::parzen_window);
  module.impl("periodic_parzen_window", &torchscience::composite::window_function::periodic_parzen_window);
  module.impl("blackman_harris_window", &torchscience::composite::window_function::blackman_harris_window);
  module.impl("periodic_blackman_harris_window", &torchscience::composite::window_function::periodic_blackman_harris_window);
  module.impl("flat_top_window", &torchscience::composite::window_function::flat_top_window);
  module.impl("periodic_flat_top_window", &torchscience::composite::window_function::periodic_flat_top_window);
  module.impl("sine_window", &torchscience::composite::window_function::sine_window);
  module.impl("periodic_sine_window", &torchscience::composite::window_function::periodic_sine_window);
  module.impl("bartlett_hann_window", &torchscience::composite::window_function::bartlett_hann_window);
  module.impl("periodic_bartlett_hann_window", &torchscience::composite::window_function::periodic_bartlett_hann_window);
  module.impl("lanczos_window", &torchscience::composite::window_function::lanczos_window);
  module.impl("periodic_lanczos_window", &torchscience::composite::window_function::periodic_lanczos_window);

  module.impl("gaussian_window", &torchscience::composite::window_function::gaussian_window);
  module.impl("periodic_gaussian_window", &torchscience::composite::window_function::periodic_gaussian_window);
  module.impl("general_hamming_window", &torchscience::composite::window_function::general_hamming_window);
  module.impl("periodic_general_hamming_window", &torchscience::composite::window_function::periodic_general_hamming_window);
  module.impl("general_cosine_window", &torchscience::composite::window_function::general_cosine_window);
  module.impl("periodic_general_cosine_window", &torchscience::composite::window_function::periodic_general_cosine_window);
  module.impl("tukey_window", &torchscience::composite::window_function::tukey_window);
  module.impl("periodic_tukey_window", &torchscience::composite::window_function::periodic_tukey_window);
  module.impl("exponential_window", &torchscience::composite::window_function::exponential_window);
  module.impl("periodic_exponential_window", &torchscience::composite::window_function::periodic_exponential_window);
  module.impl("hann_poisson_window", &torchscience::composite::window_function::hann_poisson_window);
  module.impl("periodic_hann_poisson_window", &torchscience::composite::window_function::periodic_hann_poisson_window);
  module.impl("generalized_normal_window", &torchscience::composite::window_function::generalized_normal_window);
  module.impl("periodic_generalized_normal_window", &torchscience::composite::window_function::periodic_generalized_normal_window);
  module.impl("kaiser_window", &torchscience::composite::window_function::kaiser_window);
  module.impl("periodic_kaiser_window", &torchscience::composite::window_function::periodic_kaiser_window);
  module.impl("planck_taper_window", &torchscience::composite::window_function::planck_taper_window);
  module.impl("periodic_planck_taper_window", &torchscience::composite::window_function::periodic_planck_taper_window);
  module.impl("planck_bessel_window", &torchscience::composite::window_function::planck_bessel_window);
  module.impl("periodic_planck_bessel_window", &torchscience::composite::window_function::periodic_planck_bessel_window);
}

// =============================================================================
// CompositeImplicitAutograd registrations (Pattern C: FFT-based windows)
// Gradients flow automatically through ATen operations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, CompositeImplicitAutograd, m) {
  // Pattern C: FFT-based windows
  m.impl("dolph_chebyshev_window", torchscience::composite::window_function::dolph_chebyshev_window);
  m.impl("periodic_dolph_chebyshev_window", torchscience::composite::window_function::periodic_dolph_chebyshev_window);
  m.impl("ultraspherical_window", torchscience::composite::window_function::ultraspherical_window);
  m.impl("periodic_ultraspherical_window", torchscience::composite::window_function::periodic_ultraspherical_window);

  // Phase 5: Eigenvalue-based and polynomial windows
  m.impl("discrete_prolate_spheroidal_sequence_window", torchscience::composite::window_function::discrete_prolate_spheroidal_sequence_window);
  m.impl("periodic_discrete_prolate_spheroidal_sequence_window", torchscience::composite::window_function::periodic_discrete_prolate_spheroidal_sequence_window);
  m.impl("approximate_confined_gaussian_window", torchscience::composite::window_function::approximate_confined_gaussian_window);
  m.impl("periodic_approximate_confined_gaussian_window", torchscience::composite::window_function::periodic_approximate_confined_gaussian_window);
  m.impl("confined_gaussian_window", torchscience::composite::window_function::confined_gaussian_window);
  m.impl("periodic_confined_gaussian_window", torchscience::composite::window_function::periodic_confined_gaussian_window);
  m.impl("generalized_adaptive_polynomial_window", torchscience::composite::window_function::generalized_adaptive_polynomial_window);
  m.impl("periodic_generalized_adaptive_polynomial_window", torchscience::composite::window_function::periodic_generalized_adaptive_polynomial_window);
}
