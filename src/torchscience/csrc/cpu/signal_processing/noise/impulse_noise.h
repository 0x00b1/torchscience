#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::cpu::signal_processing::noise {

inline at::Tensor impulse_noise(
    at::IntArrayRef size,
    const at::Tensor& p_salt,
    const at::Tensor& p_pepper,
    double salt_value,
    double pepper_value,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<at::Generator> generator) {
  // Validate size
  TORCH_CHECK(!size.empty(), "impulse_noise: size must not be empty");
  for (auto s : size) {
    TORCH_CHECK(s >= 0, "impulse_noise: size must be non-negative, got ", s);
  }

  // Determine dtype - default to float32
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

  // Determine device
  at::Device target_device = device.value_or(at::kCPU);
  if (p_salt.defined() && p_salt.numel() > 0) {
    target_device = device.value_or(p_salt.device());
  }

  // Set options
  at::TensorOptions options;
  options = options.dtype(target_dtype);
  options = options.layout(layout.value_or(at::kStrided));
  options = options.device(target_device);

  // Expand probability tensors to match size
  at::Tensor p_salt_exp;
  at::Tensor p_pepper_exp;

  if (p_salt.dim() == 0) {
    p_salt_exp = p_salt.expand(size.vec()).contiguous();
  } else {
    p_salt_exp = p_salt.broadcast_to(size.vec()).contiguous();
  }

  if (p_pepper.dim() == 0) {
    p_pepper_exp = p_pepper.expand(size.vec()).contiguous();
  } else {
    p_pepper_exp = p_pepper.broadcast_to(size.vec()).contiguous();
  }

  // Move to target device and ensure float for comparison
  p_salt_exp = p_salt_exp.to(target_device).to(at::kFloat);
  p_pepper_exp = p_pepper_exp.to(target_device).to(at::kFloat);

  // Generate uniform random numbers for selection
  at::Tensor uniform = at::rand(size.vec(), generator, options.dtype(at::kFloat));

  // Create output tensor initialized to zero
  at::Tensor result = at::zeros(size.vec(), options);

  // Apply salt and pepper noise:
  // - If uniform < p_pepper: set to pepper_value
  // - If uniform > 1 - p_salt: set to salt_value
  // - Otherwise: keep as 0

  // Pepper mask: uniform < p_pepper
  at::Tensor pepper_mask = uniform < p_pepper_exp;

  // Salt mask: uniform > (1 - p_salt)
  at::Tensor salt_mask = uniform > (1.0 - p_salt_exp);

  // Apply pepper first, then salt (salt takes precedence in overlap)
  result = at::where(pepper_mask, at::full_like(result, pepper_value), result);
  result = at::where(salt_mask, at::full_like(result, salt_value), result);

  return result;
}

}  // namespace torchscience::cpu::signal_processing::noise

TORCH_LIBRARY_IMPL(torchscience, CompositeExplicitAutograd, module) {
  module.impl(
      "impulse_noise",
      &torchscience::cpu::signal_processing::noise::impulse_noise);
}
