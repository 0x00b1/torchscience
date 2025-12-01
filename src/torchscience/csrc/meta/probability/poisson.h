#pragma once

#include <ATen/ATen.h>
#include <torch/library.h>

namespace torchscience::meta::probability {

at::Tensor poisson_cumulative_distribution(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto result_shape = at::infer_size(k.sizes(), rate.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor> poisson_cumulative_distribution_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(rate));
}

at::Tensor poisson_probability_mass(
    const at::Tensor& k,
    const at::Tensor& rate) {
  auto result_shape = at::infer_size(k.sizes(), rate.sizes());
  return at::empty(result_shape, k.options());
}

std::tuple<at::Tensor, at::Tensor> poisson_probability_mass_backward(
    const at::Tensor& grad,
    const at::Tensor& k,
    const at::Tensor& rate) {
  return std::make_tuple(
      at::empty_like(k),
      at::empty_like(rate));
}

TORCH_LIBRARY_IMPL(torchscience, Meta, m) {
  m.impl("poisson_cumulative_distribution", &poisson_cumulative_distribution);
  m.impl("poisson_cumulative_distribution_backward", &poisson_cumulative_distribution_backward);
  m.impl("poisson_probability_mass", &poisson_probability_mass);
  m.impl("poisson_probability_mass_backward", &poisson_probability_mass_backward);
}

}  // namespace torchscience::meta::probability
