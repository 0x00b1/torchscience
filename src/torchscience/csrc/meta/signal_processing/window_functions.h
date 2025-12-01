#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::window_function {

namespace {

inline at::TensorOptions build_meta_options(
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout
) {
  return at::TensorOptions()
    .dtype(dtype.value_or(c10::typeMetaToScalarType(at::get_default_dtype())))
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
}

}  // anonymous namespace

// =============================================================================
// Parameterless windows - all have the same signature
// =============================================================================

#define TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(name)                     \
inline at::Tensor name##_window(                                                \
  int64_t n,                                                                    \
  c10::optional<at::ScalarType> dtype,                                          \
  c10::optional<at::Layout> layout,                                             \
  c10::optional<at::Device> device,                                             \
  bool requires_grad                                                            \
) {                                                                             \
  (void)device;  /* Meta ignores device */                                      \
  auto options = build_meta_options(dtype, layout);                             \
  return at::empty({n}, options).requires_grad_(requires_grad);                 \
}

TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(hamming)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_hamming)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(blackman)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_blackman)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(bartlett)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_bartlett)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(cosine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_cosine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(nuttall)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_nuttall)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(triangular)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_triangular)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(welch)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_welch)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(parzen)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_parzen)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(blackman_harris)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_blackman_harris)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(flat_top)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_flat_top)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(sine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_sine)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(bartlett_hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_bartlett_hann)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(lanczos)
TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW(periodic_lanczos)

#undef TORCHSCIENCE_DEFINE_META_PARAMETERLESS_WINDOW

// =============================================================================
// Parameterized windows: Gaussian
// =============================================================================

inline at::Tensor gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(std_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_gaussian_window(
  int64_t n,
  const at::Tensor& std_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return gaussian_window(n, std_input, dtype, layout, device);
}

inline at::Tensor gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(std_input, at::TensorOptions().dtype(std_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_gaussian_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& std_input
) {
  return gaussian_window_backward(grad_output, output, n, std_input);
}

// =============================================================================
// Parameterized windows: General Hamming
// =============================================================================

inline at::Tensor general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_general_hamming_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_hamming_window(n, alpha_input, dtype, layout, device);
}

inline at::Tensor general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(alpha_input, at::TensorOptions().dtype(alpha_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_general_hamming_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return general_hamming_window_backward(grad_output, output, n, alpha_input);
}

// =============================================================================
// Parameterized windows: General Cosine
// =============================================================================

inline at::Tensor general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(coeffs_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_general_cosine_window(
  int64_t n,
  const at::Tensor& coeffs_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return general_cosine_window(n, coeffs_input, dtype, layout, device);
}

inline at::Tensor general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(coeffs_input, at::TensorOptions().dtype(coeffs_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_general_cosine_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& coeffs_input
) {
  return general_cosine_window_backward(grad_output, output, n, coeffs_input);
}

// =============================================================================
// Parameterized windows: Tukey
// =============================================================================

inline at::Tensor tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_tukey_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return tukey_window(n, alpha_input, dtype, layout, device);
}

inline at::Tensor tukey_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(alpha_input, at::TensorOptions().dtype(alpha_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_tukey_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return tukey_window_backward(grad_output, output, n, alpha_input);
}

// =============================================================================
// Parameterized windows: Exponential
// =============================================================================

inline at::Tensor exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(tau_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_exponential_window(
  int64_t n,
  const at::Tensor& tau_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return exponential_window(n, tau_input, dtype, layout, device);
}

inline at::Tensor exponential_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& tau_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(tau_input, at::TensorOptions().dtype(tau_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_exponential_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& tau_input
) {
  return exponential_window_backward(grad_output, output, n, tau_input);
}

// =============================================================================
// Parameterized windows: Hann-Poisson
// =============================================================================

inline at::Tensor hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(alpha_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_hann_poisson_window(
  int64_t n,
  const at::Tensor& alpha_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return hann_poisson_window(n, alpha_input, dtype, layout, device);
}

inline at::Tensor hann_poisson_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(alpha_input, at::TensorOptions().dtype(alpha_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_hann_poisson_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& alpha_input
) {
  return hann_poisson_window_backward(grad_output, output, n, alpha_input);
}

// =============================================================================
// Parameterized windows: Generalized Normal (two parameters)
// =============================================================================

inline at::Tensor generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto promoted = at::result_type(p_input, sigma_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_generalized_normal_window(
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return generalized_normal_window(n, p_input, sigma_input, dtype, layout, device);
}

inline std::tuple<at::Tensor, at::Tensor> generalized_normal_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  auto grad_p = at::empty_like(p_input, at::TensorOptions().dtype(p_input.scalar_type()).device(at::kMeta));
  auto grad_sigma = at::empty_like(sigma_input, at::TensorOptions().dtype(sigma_input.scalar_type()).device(at::kMeta));
  return std::make_tuple(grad_p, grad_sigma);
}

inline std::tuple<at::Tensor, at::Tensor> periodic_generalized_normal_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& p_input,
  const at::Tensor& sigma_input
) {
  return generalized_normal_window_backward(grad_output, output, n, p_input, sigma_input);
}

// =============================================================================
// Parameterized windows: Kaiser
// =============================================================================

inline at::Tensor kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(beta_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_kaiser_window(
  int64_t n,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return kaiser_window(n, beta_input, dtype, layout, device);
}

inline at::Tensor kaiser_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& beta_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(beta_input, at::TensorOptions().dtype(beta_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_kaiser_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& beta_input
) {
  return kaiser_window_backward(grad_output, output, n, beta_input);
}

// =============================================================================
// Parameterized windows: Planck-taper
// =============================================================================

inline at::Tensor planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto out_dtype = dtype.value_or(epsilon_input.scalar_type());
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_planck_taper_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_taper_window(n, epsilon_input, dtype, layout, device);
}

inline at::Tensor planck_taper_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  return at::empty_like(epsilon_input, at::TensorOptions().dtype(epsilon_input.scalar_type()).device(at::kMeta));
}

inline at::Tensor periodic_planck_taper_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input
) {
  return planck_taper_window_backward(grad_output, output, n, epsilon_input);
}

// =============================================================================
// Parameterized windows: Planck-Bessel (two parameters)
// =============================================================================

inline at::Tensor planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  (void)device;
  auto promoted = at::result_type(epsilon_input, beta_input);
  auto out_dtype = dtype.value_or(promoted);
  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(layout.value_or(at::kStrided))
    .device(at::kMeta);
  return at::empty({n}, options);
}

inline at::Tensor periodic_planck_bessel_window(
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input,
  c10::optional<at::ScalarType> dtype,
  c10::optional<at::Layout> layout,
  c10::optional<at::Device> device
) {
  return planck_bessel_window(n, epsilon_input, beta_input, dtype, layout, device);
}

inline std::tuple<at::Tensor, at::Tensor> planck_bessel_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input
) {
  (void)grad_output;
  (void)output;
  (void)n;
  auto grad_epsilon = at::empty_like(epsilon_input, at::TensorOptions().dtype(epsilon_input.scalar_type()).device(at::kMeta));
  auto grad_beta = at::empty_like(beta_input, at::TensorOptions().dtype(beta_input.scalar_type()).device(at::kMeta));
  return std::make_tuple(grad_epsilon, grad_beta);
}

inline std::tuple<at::Tensor, at::Tensor> periodic_planck_bessel_window_backward(
  const at::Tensor& grad_output,
  const at::Tensor& output,
  int64_t n,
  const at::Tensor& epsilon_input,
  const at::Tensor& beta_input
) {
  return planck_bessel_window_backward(grad_output, output, n, epsilon_input, beta_input);
}

}  // namespace torchscience::meta::window_function

// =============================================================================
// TORCH_LIBRARY_IMPL registrations
// =============================================================================

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("hann_window", torchscience::meta::window_function::hann_window);
  m.impl("periodic_hann_window", torchscience::meta::window_function::periodic_hann_window);
  m.impl("hamming_window", torchscience::meta::window_function::hamming_window);
  m.impl("periodic_hamming_window", torchscience::meta::window_function::periodic_hamming_window);
  m.impl("blackman_window", torchscience::meta::window_function::blackman_window);
  m.impl("periodic_blackman_window", torchscience::meta::window_function::periodic_blackman_window);
  m.impl("bartlett_window", torchscience::meta::window_function::bartlett_window);
  m.impl("periodic_bartlett_window", torchscience::meta::window_function::periodic_bartlett_window);
  m.impl("cosine_window", torchscience::meta::window_function::cosine_window);
  m.impl("periodic_cosine_window", torchscience::meta::window_function::periodic_cosine_window);
  m.impl("nuttall_window", torchscience::meta::window_function::nuttall_window);
  m.impl("periodic_nuttall_window", torchscience::meta::window_function::periodic_nuttall_window);
  m.impl("triangular_window", torchscience::meta::window_function::triangular_window);
  m.impl("periodic_triangular_window", torchscience::meta::window_function::periodic_triangular_window);
  m.impl("welch_window", torchscience::meta::window_function::welch_window);
  m.impl("periodic_welch_window", torchscience::meta::window_function::periodic_welch_window);
  m.impl("parzen_window", torchscience::meta::window_function::parzen_window);
  m.impl("periodic_parzen_window", torchscience::meta::window_function::periodic_parzen_window);
  m.impl("blackman_harris_window", torchscience::meta::window_function::blackman_harris_window);
  m.impl("periodic_blackman_harris_window", torchscience::meta::window_function::periodic_blackman_harris_window);
  m.impl("flat_top_window", torchscience::meta::window_function::flat_top_window);
  m.impl("periodic_flat_top_window", torchscience::meta::window_function::periodic_flat_top_window);
  m.impl("sine_window", torchscience::meta::window_function::sine_window);
  m.impl("periodic_sine_window", torchscience::meta::window_function::periodic_sine_window);
  m.impl("bartlett_hann_window", torchscience::meta::window_function::bartlett_hann_window);
  m.impl("periodic_bartlett_hann_window", torchscience::meta::window_function::periodic_bartlett_hann_window);
  m.impl("lanczos_window", torchscience::meta::window_function::lanczos_window);
  m.impl("periodic_lanczos_window", torchscience::meta::window_function::periodic_lanczos_window);

  m.impl("gaussian_window", torchscience::meta::window_function::gaussian_window);
  m.impl("periodic_gaussian_window", torchscience::meta::window_function::periodic_gaussian_window);
  m.impl("gaussian_window_backward", torchscience::meta::window_function::gaussian_window_backward);
  m.impl("periodic_gaussian_window_backward", torchscience::meta::window_function::periodic_gaussian_window_backward);

  m.impl("general_hamming_window", torchscience::meta::window_function::general_hamming_window);
  m.impl("periodic_general_hamming_window", torchscience::meta::window_function::periodic_general_hamming_window);
  m.impl("general_hamming_window_backward", torchscience::meta::window_function::general_hamming_window_backward);
  m.impl("periodic_general_hamming_window_backward", torchscience::meta::window_function::periodic_general_hamming_window_backward);

  m.impl("general_cosine_window", torchscience::meta::window_function::general_cosine_window);
  m.impl("periodic_general_cosine_window", torchscience::meta::window_function::periodic_general_cosine_window);
  m.impl("general_cosine_window_backward", torchscience::meta::window_function::general_cosine_window_backward);
  m.impl("periodic_general_cosine_window_backward", torchscience::meta::window_function::periodic_general_cosine_window_backward);

  m.impl("tukey_window", torchscience::meta::window_function::tukey_window);
  m.impl("periodic_tukey_window", torchscience::meta::window_function::periodic_tukey_window);
  m.impl("tukey_window_backward", torchscience::meta::window_function::tukey_window_backward);
  m.impl("periodic_tukey_window_backward", torchscience::meta::window_function::periodic_tukey_window_backward);

  m.impl("exponential_window", torchscience::meta::window_function::exponential_window);
  m.impl("periodic_exponential_window", torchscience::meta::window_function::periodic_exponential_window);
  m.impl("exponential_window_backward", torchscience::meta::window_function::exponential_window_backward);
  m.impl("periodic_exponential_window_backward", torchscience::meta::window_function::periodic_exponential_window_backward);

  m.impl("hann_poisson_window", torchscience::meta::window_function::hann_poisson_window);
  m.impl("periodic_hann_poisson_window", torchscience::meta::window_function::periodic_hann_poisson_window);
  m.impl("hann_poisson_window_backward", torchscience::meta::window_function::hann_poisson_window_backward);
  m.impl("periodic_hann_poisson_window_backward", torchscience::meta::window_function::periodic_hann_poisson_window_backward);

  m.impl("generalized_normal_window", torchscience::meta::window_function::generalized_normal_window);
  m.impl("periodic_generalized_normal_window", torchscience::meta::window_function::periodic_generalized_normal_window);
  m.impl("generalized_normal_window_backward", torchscience::meta::window_function::generalized_normal_window_backward);
  m.impl("periodic_generalized_normal_window_backward", torchscience::meta::window_function::periodic_generalized_normal_window_backward);

  m.impl("kaiser_window", torchscience::meta::window_function::kaiser_window);
  m.impl("periodic_kaiser_window", torchscience::meta::window_function::periodic_kaiser_window);
  m.impl("kaiser_window_backward", torchscience::meta::window_function::kaiser_window_backward);
  m.impl("periodic_kaiser_window_backward", torchscience::meta::window_function::periodic_kaiser_window_backward);

  m.impl("planck_taper_window", torchscience::meta::window_function::planck_taper_window);
  m.impl("periodic_planck_taper_window", torchscience::meta::window_function::periodic_planck_taper_window);
  m.impl("planck_taper_window_backward", torchscience::meta::window_function::planck_taper_window_backward);
  m.impl("periodic_planck_taper_window_backward", torchscience::meta::window_function::periodic_planck_taper_window_backward);

  m.impl("planck_bessel_window", torchscience::meta::window_function::planck_bessel_window);
  m.impl("periodic_planck_bessel_window", torchscience::meta::window_function::periodic_planck_bessel_window);
  m.impl("planck_bessel_window_backward", torchscience::meta::window_function::planck_bessel_window_backward);
  m.impl("periodic_planck_bessel_window_backward", torchscience::meta::window_function::periodic_planck_bessel_window_backward);
}
