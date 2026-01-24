#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::color {

/**
 * Meta implementation for Oklch to sRGB shape inference.
 */
inline at::Tensor oklch_to_srgb(const at::Tensor& input) {
  TORCH_CHECK(input.size(-1) == 3, "oklch_to_srgb: input must have last dimension 3, got ", input.size(-1));
  return at::empty_like(input);
}

/**
 * Meta implementation for Oklch to sRGB backward shape inference.
 */
inline at::Tensor oklch_to_srgb_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  return at::empty_like(input);
}

}  // namespace torchscience::meta::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("oklch_to_srgb", &torchscience::meta::graphics::color::oklch_to_srgb);
  m.impl("oklch_to_srgb_backward", &torchscience::meta::graphics::color::oklch_to_srgb_backward);
}
