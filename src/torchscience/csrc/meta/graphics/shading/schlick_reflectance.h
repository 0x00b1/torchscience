#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::shading {

/**
 * Meta implementation for Schlick reflectance shape inference.
 */
inline at::Tensor schlick_reflectance(
    const at::Tensor& cosine,
    const at::Tensor& r0
) {
  // Compute broadcast shape
  auto broadcasted = at::broadcast_tensors({cosine, r0});
  return at::empty_like(broadcasted[0]);
}

/**
 * Meta implementation for Schlick reflectance backward shape inference.
 */
inline at::Tensor schlick_reflectance_backward(
    const at::Tensor& grad_output,
    const at::Tensor& cosine,
    const at::Tensor& r0
) {
  auto broadcasted = at::broadcast_tensors({grad_output, cosine, r0});
  return at::empty_like(broadcasted[1]);
}

}  // namespace torchscience::meta::graphics::shading

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("schlick_reflectance", &torchscience::meta::graphics::shading::schlick_reflectance);
  m.impl("schlick_reflectance_backward", &torchscience::meta::graphics::shading::schlick_reflectance_backward);
}
