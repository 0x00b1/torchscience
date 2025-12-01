#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::signal_processing::noise {

inline at::Tensor poisson_noise(
    at::IntArrayRef size,
    const at::Tensor& rate,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<at::Generator> generator) {
  // Validate size
  TORCH_CHECK(!size.empty(), "poisson_noise: size must not be empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "poisson_noise: size must be non-negative, got ", s);
  }

  // Check for empty tensor
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }
  if (numel == 0) {
    at::TensorOptions options;
    options = options.dtype(dtype.value_or(at::kLong));
    options = options.layout(layout.value_or(at::kStrided));
    options = options.device(device.value_or(at::kCPU));
    return at::empty(size.vec(), options);
  }

  // Determine device from rate tensor if not specified
  at::Device target_device = device.value_or(rate.device());

  // Create rate tensor with proper shape for broadcasting
  // torch.poisson expects rate tensor, not scalar
  at::Tensor rate_expanded;
  if (rate.dim() == 0) {
    // Scalar rate - expand to full size
    rate_expanded = rate.expand(size.vec()).contiguous();
  } else {
    // Tensor rate - broadcast with size
    // First, create a tensor of the target shape, then broadcast
    rate_expanded = rate.broadcast_to(size.vec()).contiguous();
  }

  // Move rate to target device
  rate_expanded = rate_expanded.to(target_device);

  // Ensure rate is float for poisson (required by PyTorch)
  if (!rate_expanded.is_floating_point()) {
    rate_expanded = rate_expanded.to(at::kFloat);
  }

  // Generate Poisson samples
  at::Tensor result = at::poisson(rate_expanded, generator);

  // Convert to requested dtype (default is int64)
  at::ScalarType target_dtype = dtype.value_or(at::kLong);
  result = result.to(target_dtype);

  return result;
}

}  // namespace torchscience::cpu::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl(
      "poisson_noise",
      &torchscience::cpu::signal_processing::noise::poisson_noise);
}
