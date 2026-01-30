#pragma once

#include <cmath>
#include <tuple>

#include "../special_functions/digamma.h"

namespace torchscience::kernel::probability {

// Gradients of gamma log probability density function
// logpdf(x; shape, scale) = (shape - 1) * log(x) - x/scale - log_gamma(shape) - shape * log(scale)
//
// d(logpdf)/dx = (shape - 1)/x - 1/scale
// d(logpdf)/dshape = log(x) - digamma(shape) - log(scale)
// d(logpdf)/dscale = x/scale^2 - shape/scale
template <typename T>
std::tuple<T, T, T> gamma_log_probability_density_backward(T gradient, T x, T shape, T scale) {
  if (x <= T(0)) {
    return {T(0), T(0), T(0)};
  }

  // d(logpdf)/dx = (shape - 1)/x - 1/scale
  T grad_x = gradient * ((shape - T(1)) / x - T(1) / scale);

  // d(logpdf)/dshape = log(x) - digamma(shape) - log(scale)
  T grad_shape = gradient * (std::log(x) - special_functions::digamma(shape) - std::log(scale));

  // d(logpdf)/dscale = x/scale^2 - shape/scale
  T grad_scale = gradient * (x / (scale * scale) - shape / scale);

  return {grad_x, grad_shape, grad_scale};
}

}  // namespace torchscience::kernel::probability
