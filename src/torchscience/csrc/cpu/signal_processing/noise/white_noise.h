#pragma once

#include <torch/extension.h>
#include <ATen/ATen.h>

namespace torchscience::cpu::signal_processing::noise {

inline at::Tensor white_noise(
  at::IntArrayRef size,
  const c10::optional<at::ScalarType> dtype,
  const c10::optional<at::Layout> layout,
  const c10::optional<at::Device> device,
  const bool requires_grad,
  const c10::optional<at::Generator> generator
) {
  TORCH_CHECK(size.size() > 0, "white_noise: size must be non-empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "white_noise: size elements must be non-negative, got ", s);
  }

  // Determine output dtype and device
  at::ScalarType out_dtype = dtype.value_or(
    c10::typeMetaToScalarType(at::get_default_dtype())
  );
  at::Device out_device = device.value_or(at::kCPU);
  at::Layout out_layout = layout.value_or(at::kStrided);

  auto options = at::TensorOptions()
    .dtype(out_dtype)
    .layout(out_layout)
    .device(out_device)
    .requires_grad(false);

  // Generate standard normal random samples
  at::Tensor result = at::randn(size.vec(), generator, options);

  // Set requires_grad if requested
  if (requires_grad) {
    result = result.requires_grad_(true);
  }

  return result;
}

} // namespace torchscience::cpu::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl("white_noise", &torchscience::cpu::signal_processing::noise::white_noise);
}
