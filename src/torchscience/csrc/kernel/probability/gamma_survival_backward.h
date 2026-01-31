#pragma once

#include <cmath>
#include <tuple>

#include "gamma_probability_density.h"
#include "../special_functions/regularized_gamma_q_backward.h"

namespace torchscience::kernel::probability {

// Gamma survival function gradient
// S(x; shape, scale) = Q(shape, x/scale)
//
// dS/dx = -pdf(x; shape, scale)  [since S = 1 - CDF and dCDF/dx = pdf]
// dS/dshape = dQ/da where a = shape
// dS/dscale = dQ/dz * dz/dscale = dQ/dz * (-x/scale^2)
template <typename T>
std::tuple<T, T, T> gamma_survival_backward(T gradient, T x, T shape, T scale) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  // Compute z = x / scale for the regularized gamma function
  T z = x / scale;

  // Get gradients of Q(shape, z) with respect to shape and z
  auto [grad_shape_raw, grad_z_raw] = special_functions::regularized_gamma_q_backward(T(1), shape, z);

  // dS/dx = -pdf(x) since S = 1 - CDF
  T pdf = gamma_probability_density(x, shape, scale);
  T grad_x = gradient * (-pdf);

  // dS/dshape from regularized_gamma_q
  T grad_shape = gradient * grad_shape_raw;

  // dS/dscale = dS/dz * dz/dscale = grad_z_raw * (-x/scale^2)
  T grad_scale = gradient * grad_z_raw * (-x / (scale * scale));

  return {grad_x, grad_shape, grad_scale};
}

}  // namespace torchscience::kernel::probability
