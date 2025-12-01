#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::signal_processing::noise {

inline at::Tensor shot_noise(
    at::IntArrayRef size,
    const at::Tensor& rate,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    bool requires_grad,
    std::optional<at::Generator> generator) {
  // Validate size
  TORCH_CHECK(!size.empty(), "shot_noise: size must not be empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "shot_noise: size must be non-negative, got ", s);
  }

  // Determine dtype - default to float32 for differentiability
  at::ScalarType target_dtype = dtype.value_or(at::kFloat);

  // Check for empty tensor
  int64_t numel = 1;
  for (auto s : size) {
    numel *= s;
  }
  if (numel == 0) {
    at::TensorOptions options;
    options = options.dtype(target_dtype);
    options = options.layout(layout.value_or(at::kStrided));
    options = options.device(device.value_or(at::kCPU));
    return at::empty(size.vec(), options);
  }

  // Determine device from rate tensor if not specified
  at::Device target_device = device.value_or(rate.device());

  // Create rate tensor with proper shape for broadcasting
  at::Tensor rate_expanded;
  if (rate.dim() == 0) {
    rate_expanded = rate.expand(size.vec()).contiguous();
  } else {
    rate_expanded = rate.broadcast_to(size.vec()).contiguous();
  }

  // Move rate to target device and ensure float for computation
  rate_expanded = rate_expanded.to(target_device);
  at::ScalarType compute_dtype = at::isFloatingType(target_dtype)
      ? target_dtype
      : at::kFloat;
  rate_expanded = rate_expanded.to(compute_dtype);

  // Set options for generating random numbers
  at::TensorOptions options;
  options = options.dtype(compute_dtype);
  options = options.layout(layout.value_or(at::kStrided));
  options = options.device(target_device);

  // Use Gaussian approximation: N(rate, sqrt(rate))
  // This is valid for all rates and differentiable.
  // For rate >= 10, this is a very good approximation.
  // For rate < 10, it's less accurate but still differentiable.
  //
  // Alternative: Gumbel-softmax relaxation for low rates, but that's
  // significantly more complex and the Gaussian approximation is often
  // sufficient for differentiable training.

  // Generate standard normal samples
  at::Tensor z = at::randn(size.vec(), generator, options);

  // Shot noise = rate + sqrt(rate) * z
  // We clamp sqrt(rate) to avoid issues with rate=0
  at::Tensor std = at::sqrt(at::clamp(rate_expanded, 0.0));
  at::Tensor result = rate_expanded + std * z;

  // Clamp to non-negative since Poisson is non-negative
  // Use soft clamp for differentiability: softplus is smooth
  // But for simplicity, we use ReLU which is subgradient differentiable
  result = at::relu(result);

  // Convert to target dtype if needed
  if (target_dtype != compute_dtype) {
    result = result.to(target_dtype);
  }

  // Set requires_grad if requested
  if (requires_grad && at::isFloatingType(target_dtype)) {
    result = result.requires_grad_(true);
  }

  return result;
}

}  // namespace torchscience::cpu::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl(
      "shot_noise",
      &torchscience::cpu::signal_processing::noise::shot_noise);
}
