#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cmath>

namespace torchscience::cpu::signal_processing::noise {

inline at::Tensor pink_noise(
  at::IntArrayRef size,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad,
  const c10::optional<at::Generator> generator
) {
  TORCH_CHECK(size.size() > 0, "pink_noise: size must be non-empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "pink_noise: size elements must be non-negative, got ", s);
  }

  // Determine output dtype and device
  at::ScalarType out_dtype = dtype.value_or(
    c10::typeMetaToScalarType(at::get_default_dtype())
  );
  at::Device out_device = device.value_or(at::kCPU);
  at::Layout out_layout = layout.value_or(at::kStrided);

  // Handle empty tensor case
  int64_t n = size.back();  // Last dimension is the sample axis
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }

  if (numel == 0 || n == 0) {
    auto options = at::TensorOptions()
      .dtype(out_dtype)
      .layout(out_layout)
      .device(out_device)
      .requires_grad(requires_grad);
    return at::empty(size.vec(), options);
  }

  // Handle n=1 special case (return zeros - can't have 1/f spectrum with 1 sample)
  if (n == 1) {
    auto options = at::TensorOptions()
      .dtype(out_dtype)
      .layout(out_layout)
      .device(out_device)
      .requires_grad(requires_grad);
    return at::zeros(size.vec(), options);
  }

  // Use float64 for computation to ensure precision
  at::ScalarType compute_dtype = at::kDouble;

  // For half-precision types, we'll compute in float32 then cast back
  if (out_dtype == at::kHalf || out_dtype == at::kBFloat16) {
    compute_dtype = at::kFloat;
  } else if (out_dtype == at::kDouble) {
    compute_dtype = at::kDouble;
  } else {
    compute_dtype = at::kFloat;
  }

  auto compute_options = at::TensorOptions()
    .dtype(compute_dtype)
    .layout(out_layout)
    .device(out_device);

  // Step 1: Generate white noise
  at::Tensor white = at::randn(size.vec(), generator, compute_options);

  // Step 2: Compute real FFT along last dimension
  at::Tensor spectrum = at::fft_rfft(white, /*n=*/c10::nullopt, /*dim=*/-1);

  // Step 3: Build frequency scaling vector S[k] = sqrt(N/k) for k > 0, S[0] = 0
  // spectrum has shape [..., N/2 + 1]
  int64_t freq_size = spectrum.size(-1);

  // Create frequency indices [0, 1, 2, ..., N/2]
  at::Tensor freq_indices = at::arange(freq_size, compute_options);

  // Scaling: sqrt(N/k) for k > 0, 0 for k = 0
  // We use sqrt(N) / sqrt(k) = sqrt(N/k)
  at::Tensor scaling = at::where(
    freq_indices > 0,
    at::sqrt(static_cast<double>(n) / freq_indices),
    at::zeros({1}, compute_options)
  );

  // Step 4: Apply scaling (broadcast over batch dimensions)
  at::Tensor shaped_spectrum = spectrum * scaling;

  // Step 5: Inverse FFT to get time-domain signal
  at::Tensor result = at::fft_irfft(shaped_spectrum, /*n=*/n, /*dim=*/-1);

  // Step 6: Normalize to approximately unit variance
  // The theoretical variance before normalization is sum(N/k) for k=1..N/2
  // which is approximately N * (ln(N/2) + gamma) where gamma is Euler's constant
  // We normalize by sqrt of this
  double harmonic_sum = 0.0;
  for (int64_t k = 1; k <= n / 2; ++k) {
    harmonic_sum += static_cast<double>(n) / static_cast<double>(k);
  }
  double norm_factor = std::sqrt(harmonic_sum / static_cast<double>(n));

  result = result / norm_factor;

  // Cast to output dtype if needed
  if (result.scalar_type() != out_dtype) {
    result = result.to(out_dtype);
  }

  // Set requires_grad if requested
  if (requires_grad) {
    result = result.requires_grad_(true);
  }

  return result;
}

} // namespace torchscience::cpu::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl("pink_noise", &torchscience::cpu::signal_processing::noise::pink_noise);
}
