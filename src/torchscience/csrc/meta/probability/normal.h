#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor normal_cumulative_distribution(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  // Infer broadcast shape
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> normal_cumulative_distribution_backward_backward(
    const at::Tensor& gg_x,
    const at::Tensor& gg_loc,
    const at::Tensor& gg_scale,
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(grad),
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_probability_density(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_quantile(
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(p.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(p, loc);
  output_dtype = at::result_type(at::empty({}, p.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, p.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_quantile_backward(
    const at::Tensor& grad,
    const at::Tensor& p,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(p),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_survival(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

at::Tensor normal_log_probability_density(
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  auto shape1 = at::infer_size(x.sizes(), loc.sizes());
  auto output_shape = at::infer_size(shape1, scale.sizes());
  auto output_dtype = at::result_type(x, loc);
  output_dtype = at::result_type(at::empty({}, x.options().dtype(output_dtype)), scale);
  return at::empty(output_shape, x.options().dtype(output_dtype));
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> normal_log_probability_density_backward(
    const at::Tensor& grad,
    const at::Tensor& x,
    const at::Tensor& loc,
    const at::Tensor& scale) {
  return std::make_tuple(
      at::empty_like(x),
      at::empty_like(loc),
      at::empty_like(scale));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("normal_cumulative_distribution", &normal_cumulative_distribution);
  m.impl("normal_cumulative_distribution_backward", &normal_cumulative_distribution_backward);
  m.impl("normal_cumulative_distribution_backward_backward", &normal_cumulative_distribution_backward_backward);
  m.impl("normal_probability_density", &normal_probability_density);
  m.impl("normal_probability_density_backward", &normal_probability_density_backward);
  m.impl("normal_quantile", &normal_quantile);
  m.impl("normal_quantile_backward", &normal_quantile_backward);
  m.impl("normal_survival", &normal_survival);
  m.impl("normal_survival_backward", &normal_survival_backward);
  m.impl("normal_log_probability_density", &normal_log_probability_density);
  m.impl("normal_log_probability_density_backward", &normal_log_probability_density_backward);
}

}  // namespace torchscience::meta::probability
