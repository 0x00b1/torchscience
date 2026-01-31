#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor binomial_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

at::Tensor binomial_probability_mass(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

at::Tensor binomial_log_probability_mass(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_log_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

at::Tensor binomial_survival(
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  auto result_shape = at::infer_size(k.sizes(), n.sizes());
  result_shape = at::infer_size(result_shape, p.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> binomial_survival_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& n,
    const at::Tensor& p) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(n),
      at::empty_like(p));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("binomial_cumulative_distribution", &binomial_cumulative_distribution);
  m.impl("binomial_cumulative_distribution_backward", &binomial_cumulative_distribution_backward);
  m.impl("binomial_log_probability_mass", &binomial_log_probability_mass);
  m.impl("binomial_log_probability_mass_backward", &binomial_log_probability_mass_backward);
  m.impl("binomial_probability_mass", &binomial_probability_mass);
  m.impl("binomial_probability_mass_backward", &binomial_probability_mass_backward);
  m.impl("binomial_survival", &binomial_survival);
  m.impl("binomial_survival_backward", &binomial_survival_backward);
}

}  // namespace torchscience::meta::probability
