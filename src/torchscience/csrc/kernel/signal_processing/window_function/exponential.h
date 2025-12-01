#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Exponential (Poisson) window: w[k] = exp(-|k - center| / tau)
// where center = denom / 2, and denom is (n-1) for symmetric, n for periodic
template<typename scalar_t>
inline scalar_t exponential(int64_t i, int64_t n, scalar_t tau, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  scalar_t center = denom / scalar_t(2);
  scalar_t abs_diff = std::abs(scalar_t(i) - center);
  return std::exp(-abs_diff / tau);
}

// Gradient w.r.t. tau parameter
// d/d(tau) exp(-|k - center| / tau) = exp(...) * |k - center| / tau^2
template<typename scalar_t>
inline scalar_t exponential_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t tau,
  bool periodic,
  scalar_t forward_value
) {
  if (n == 1 || tau == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t center = denom / scalar_t(2);
  scalar_t abs_diff = std::abs(scalar_t(i) - center);

  // d/d(tau) exp(-|k - center| / tau) = exp(...) * |k - center| / tau^2
  return grad_out * forward_value * abs_diff / (tau * tau);
}

}  // namespace torchscience::kernel::window_function
