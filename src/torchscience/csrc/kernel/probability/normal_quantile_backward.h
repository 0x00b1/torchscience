#pragma once

#include <cmath>
#include "normal_quantile.h"

namespace torchscience::kernel::probability {

// Gradients of normal PPF (quantile function)
//
// ppf(p; loc, scale) = loc + scale * z
// where z = sqrt(2) * erfinv(2*p - 1)
//
// d(ppf)/dp = scale / pdf(ppf(p))
//           = scale * sqrt(2*pi) * exp(z^2/2)
// d(ppf)/dloc = 1
// d(ppf)/dscale = z
template <typename T>
inline T sqrt_2pi_quantile() {
  return T(2.5066282746310002);  // sqrt(2 * pi)
}

template <typename T>
void normal_quantile_backward(
    T grad_output, T p, T loc, T scale,
    T& grad_p, T& grad_loc, T& grad_scale) {

  T z = standard_normal_quantile(p);

  // d(ppf)/dp = scale * sqrt(2*pi) * exp(z^2/2)
  // This equals 1/pdf at the quantile point
  T dppf_dp = scale * sqrt_2pi_quantile<T>() * std::exp(T(0.5) * z * z);

  grad_p = grad_output * dppf_dp;
  grad_loc = grad_output;  // d(ppf)/dloc = 1
  grad_scale = grad_output * z;  // d(ppf)/dscale = z
}

}  // namespace torchscience::kernel::probability
