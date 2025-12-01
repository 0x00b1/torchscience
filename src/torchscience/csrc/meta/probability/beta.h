#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor beta_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(x.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(a),
      at::empty_like(b));
}

at::Tensor beta_probability_density(
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(x.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(a),
      at::empty_like(b));
}

at::Tensor beta_quantile(
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  auto result_shape = at::infer_size(p.sizes(), a.sizes());
  result_shape = at::infer_size(result_shape, b.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> beta_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& a,
    const at::Tensor& b) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(a),
      at::empty_like(b));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("beta_cumulative_distribution", &beta_cumulative_distribution);
  m.impl("beta_cumulative_distribution_backward", &beta_cumulative_distribution_backward);
  m.impl("beta_probability_density", &beta_probability_density);
  m.impl("beta_probability_density_backward", &beta_probability_density_backward);
  m.impl("beta_quantile", &beta_quantile);
  m.impl("beta_quantile_backward", &beta_quantile_backward);
}

}  // namespace torchscience::meta::probability
