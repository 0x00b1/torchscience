#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Gaussian window: w[k] = exp(-0.5 * ((k - center) / (std * center))^2)
// where center = denom / 2
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t gaussian(int64_t i, int64_t n, scalar_t std_val, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t center = denom / scalar_t(2);
  scalar_t sigma = std_val * center;
  if (sigma == scalar_t(0)) {
    return (scalar_t(i) == center) ? scalar_t(1) : scalar_t(0);
  }
  scalar_t x = (scalar_t(i) - center) / sigma;
  return std::exp(scalar_t(-0.5) * x * x);
}

// Gradient w.r.t. std parameter
// d/d(std) = forward_value * x^2 / std
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t gaussian_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t std_val,
  bool periodic,
  scalar_t forward_value
) {
  if (n == 1 || std_val == scalar_t(0)) {
    return scalar_t(0);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }
  scalar_t center = denom / scalar_t(2);
  scalar_t sigma = std_val * center;
  scalar_t x = (scalar_t(i) - center) / sigma;
  return grad_out * forward_value * x * x / std_val;
}

}  // namespace torchscience::kernel::window_function
