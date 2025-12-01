#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor chi2_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& df) {
  // Infer broadcast shape
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

at::Tensor chi2_probability_density(
    const at::Tensor& x,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

at::Tensor chi2_quantile(
    const at::Tensor& p,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(p.sizes(), df.sizes());
  auto output_dtype = at::result_type(p, df);
  return at::empty(output_shape, p.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(df));
}

at::Tensor chi2_survival(
    const at::Tensor& x,
    const at::Tensor& df) {
  auto output_shape = at::infer_size(x.sizes(), df.sizes());
  auto output_dtype = at::result_type(x, df);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor> chi2_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& df) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(df));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("chi2_cumulative_distribution", &chi2_cumulative_distribution);
  m.impl("chi2_cumulative_distribution_backward", &chi2_cumulative_distribution_backward);
  m.impl("chi2_probability_density", &chi2_probability_density);
  m.impl("chi2_probability_density_backward", &chi2_probability_density_backward);
  m.impl("chi2_quantile", &chi2_quantile);
  m.impl("chi2_quantile_backward", &chi2_quantile_backward);
  m.impl("chi2_survival", &chi2_survival);
  m.impl("chi2_survival_backward", &chi2_survival_backward);
}

}  // namespace torchscience::meta::probability
