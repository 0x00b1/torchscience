#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Gradients of normal log probability density function
// logpdf(x; loc, scale) = -0.5*log(2*pi) - log(scale) - 0.5*z^2
// where z = (x - loc) / scale
//
// d(logpdf)/dx = -z / scale
// d(logpdf)/dloc = z / scale
// d(logpdf)/dscale = (z^2 - 1) / scale
template <typename T>
void normal_log_probability_density_backward(
    T grad_output, T x, T loc, T scale,
    T& grad_x, T& grad_loc, T& grad_scale) {

  T z = (x - loc) / scale;

  // d(logpdf)/dx = -z / scale
  grad_x = grad_output * (-z / scale);
  // d(logpdf)/dloc = z / scale
  grad_loc = grad_output * (z / scale);
  // d(logpdf)/dscale = (z^2 - 1) / scale
  grad_scale = grad_output * (z * z - T(1)) / scale;
}

}  // namespace torchscience::kernel::probability
