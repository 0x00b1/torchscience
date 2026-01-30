#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor gamma_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(x.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(shape),
      at::empty_like(scale));
}

at::Tensor gamma_probability_density(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(x.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(shape),
      at::empty_like(scale));
}

at::Tensor gamma_quantile(
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(p.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(shape),
      at::empty_like(scale));
}

at::Tensor gamma_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  auto result_shape = at::infer_size(x.sizes(), shape.sizes());
  result_shape = at::infer_size(result_shape, scale.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> gamma_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& shape,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(shape),
      at::empty_like(scale));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("gamma_cumulative_distribution", &gamma_cumulative_distribution);
  m.impl("gamma_cumulative_distribution_backward", &gamma_cumulative_distribution_backward);
  m.impl("gamma_probability_density", &gamma_probability_density);
  m.impl("gamma_probability_density_backward", &gamma_probability_density_backward);
  m.impl("gamma_quantile", &gamma_quantile);
  m.impl("gamma_quantile_backward", &gamma_quantile_backward);
  m.impl("gamma_log_probability_density", &gamma_log_probability_density);
  m.impl("gamma_log_probability_density_backward", &gamma_log_probability_density_backward);
}

}  // namespace torchscience::meta::probability
