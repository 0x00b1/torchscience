#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for vector reflection shape inference.
 */
inline at::Tensor reflect(const at::Tensor& direction, const at::Tensor& normal) {
  TORCH_CHECK(direction.size(-1) == 3, "reflect: direction must have last dimension 3, got ", direction.size(-1));
  TORCH_CHECK(normal.size(-1) == 3, "reflect: normal must have last dimension 3, got ", normal.size(-1));
  return at::empty_like(direction);
}

/**
 * Meta implementation for vector reflection backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor> reflect_backward(
    const at::Tensor& grad_output,
    const at::Tensor& direction,
    const at::Tensor& normal
) {
  return std::make_tuple(at::empty_like(direction), at::empty_like(normal));
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("reflect", &torchscience::meta::geometry::transform::reflect);
  m.impl("reflect_backward", &torchscience::meta::geometry::transform::reflect_backward);
}
