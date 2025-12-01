#pragma once

#include <cmath>
#include <tuple>

#include "gamma_probability_density.h"
#include "../special_functions/regularized_gamma_p_backward.h"

namespace torchscience::kernel::probability {

// Gamma CDF gradient
// F(x; shape, scale) = P(shape, x/scale)
//
// dF/dx = pdf(x)
// dF/dshape = dP/da where a = shape, using regularized_gamma_p_backward
// dF/dscale = dP/dz * dz/dscale = dP/dz * (-x/scale^2)
template <typename T>
std::tuple<T, T, T> gamma_cumulative_distribution_backward(T gradient, T x, T shape, T scale) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  // For dP/dshape and dP/dz we use the regularized_gamma_p_backward
  // It returns (grad_a, grad_x) = (grad_shape, grad_z)
  T z = x / scale;
  auto [grad_shape_raw, grad_z_raw] = special_functions::regularized_gamma_p_backward(T(1), shape, z);

  // pdf = dF/dx
  T pdf = gamma_probability_density(x, shape, scale);

  // dF/dx = pdf
  T grad_x = gradient * pdf;

  // dF/dshape from regularized_gamma_p
  T grad_shape = gradient * grad_shape_raw;

  // dF/dscale = dF/dz * dz/dscale = grad_z_raw * (-x/scale^2)
  T grad_scale = gradient * grad_z_raw * (-x / (scale * scale));

  return {grad_x, grad_shape, grad_scale};
}

}  // namespace torchscience::kernel::probability
