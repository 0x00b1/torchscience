#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::color {

/**
 * Meta implementation for sRGB to linear sRGB shape inference.
 */
inline at::Tensor srgb_to_srgb_linear(const at::Tensor& input) {
  return at::empty_like(input);
}

/**
 * Meta implementation for sRGB to linear sRGB backward shape inference.
 */
inline at::Tensor srgb_to_srgb_linear_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  return at::empty_like(input);
}

}  // namespace torchscience::meta::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("srgb_to_srgb_linear", &torchscience::meta::graphics::color::srgb_to_srgb_linear);
  m.impl("srgb_to_srgb_linear_backward", &torchscience::meta::graphics::color::srgb_to_srgb_linear_backward);
}
