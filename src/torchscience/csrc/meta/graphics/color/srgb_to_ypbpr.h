#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::graphics::color {

/**
 * Meta implementation for sRGB to YPbPr shape inference.
 */
inline at::Tensor srgb_to_ypbpr(const at::Tensor& input) {
  TORCH_CHECK(input.size(-1) == 3, "srgb_to_ypbpr: input must have last dimension 3, got ", input.size(-1));
  return at::empty_like(input);
}

/**
 * Meta implementation for sRGB to YPbPr backward shape inference.
 */
inline at::Tensor srgb_to_ypbpr_backward(
    const at::Tensor& grad_output,
    const at::Tensor& input
) {
  return at::empty_like(input);
}

}  // namespace torchscience::meta::graphics::color

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("srgb_to_ypbpr", &torchscience::meta::graphics::color::srgb_to_ypbpr);
  m.impl("srgb_to_ypbpr_backward", &torchscience::meta::graphics::color::srgb_to_ypbpr_backward);
}
