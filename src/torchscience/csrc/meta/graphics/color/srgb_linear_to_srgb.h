#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::color {

/**
 * Meta implementation for linear sRGB to sRGB shape inference.
 */
inline at::Tensor srgb_linear_to_srgb(const at::Tensor& input) {
  return at::empty_like(input);
}

/**
 * Meta implementation for linear sRGB to sRGB backward shape inference.
 */
inline at::Tensor srgb_linear_to_srgb_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  return at::empty_like(input);
}

}  // namespace torchscience::meta::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("srgb_linear_to_srgb", &torchscience::meta::graphics::color::srgb_linear_to_srgb);
  m.impl("srgb_linear_to_srgb_backward", &torchscience::meta::graphics::color::srgb_linear_to_srgb_backward);
}
