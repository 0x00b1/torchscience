#pragma once

#include <ATen/core/Tensor.h>
#include <torch/library.h>

namespace torchscience::meta::geometry::transform {

/**
 * Meta implementation for vector refraction shape inference.
 */
inline at::Tensor refract(
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& eta
) {
  TORCH_CHECK(direction.size(-1) == 3,
              "refract: direction must have last dimension 3, got ", direction.size(-1));
  TORCH_CHECK(normal.size(-1) == 3,
              "refract: normal must have last dimension 3, got ", normal.size(-1));
  return at::empty_like(direction);
}

/**
 * Meta implementation for vector refraction backward shape inference.
 */
inline std::tuple<at::Tensor, at::Tensor, at::Tensor> refract_backward(
    const at::Tensor& grad_output,
    const at::Tensor& direction,
    const at::Tensor& normal,
    const at::Tensor& eta
) {
  const int64_t num_vectors = direction.numel() / 3;
  const bool scalar_eta = (eta.numel() == 1);

  auto grad_eta = scalar_eta ? at::empty({1}, eta.options()) : at::empty_like(eta);

  return std::make_tuple(at::empty_like(direction), at::empty_like(normal), grad_eta);
}

}  // namespace torchscience::meta::geometry::transform

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("refract", &torchscience::meta::geometry::transform::refract);
  m.impl("refract_backward", &torchscience::meta::geometry::transform::refract_backward);
}
