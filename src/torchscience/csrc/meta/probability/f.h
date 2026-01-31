#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor f_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_probability_density(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_quantile(
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(p.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, p.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_survival(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

at::Tensor f_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  auto result_shape = at::infer_size(x.sizes(), dfn.sizes());
  result_shape = at::infer_size(result_shape, dfd.sizes());
  return at::empty(result_shape, x.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> f_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& dfn,
    const at::Tensor& dfd) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(dfn),
      at::empty_like(dfd));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("f_cumulative_distribution", &f_cumulative_distribution);
  m.impl("f_cumulative_distribution_backward", &f_cumulative_distribution_backward);
  m.impl("f_probability_density", &f_probability_density);
  m.impl("f_probability_density_backward", &f_probability_density_backward);
  m.impl("f_quantile", &f_quantile);
  m.impl("f_quantile_backward", &f_quantile_backward);
  m.impl("f_survival", &f_survival);
  m.impl("f_survival_backward", &f_survival_backward);
  m.impl("f_log_probability_density", &f_log_probability_density);
  m.impl("f_log_probability_density_backward", &f_log_probability_density_backward);
}

}  // namespace torchscience::meta::probability
