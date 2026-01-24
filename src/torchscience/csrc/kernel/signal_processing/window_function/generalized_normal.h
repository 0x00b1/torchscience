#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Generalized normal window: w[k] = exp(-|(k - center) / sigma|^p)
// where center = denom / 2, and denom is (n-1) for symmetric, n for periodic
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t generalized_normal(int64_t i, int64_t n, scalar_t p, scalar_t sigma, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  scalar_t center = denom / scalar_t(2);
  scalar_t x = std::abs(scalar_t(i) - center);

  if (sigma == scalar_t(0)) {
    // At center, value is 1; elsewhere undefined, return 0
    return (x == scalar_t(0)) ? scalar_t(1) : scalar_t(0);
  }

  scalar_t normalized = x / sigma;
  if (normalized == scalar_t(0)) {
    return scalar_t(1);  // exp(0) = 1
  }

  return std::exp(-std::pow(normalized, p));
}

// Gradient w.r.t. p parameter
// d/d(p) exp(-|x/sigma|^p) = exp(...) * (-|x/sigma|^p * ln(|x/sigma|))
//                          = forward_value * (-normalized^p * ln(normalized))
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t generalized_normal_backward_p(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t p,
  scalar_t sigma,
  bool periodic,
  scalar_t forward_value
) {
  (void)p;  // p not needed in this formula after simplification

  if (n == 1 || sigma == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t center = denom / scalar_t(2);
  scalar_t x = std::abs(scalar_t(i) - center);
  scalar_t normalized = x / sigma;

  if (normalized <= scalar_t(0)) {
    // At center or for numerical stability, gradient w.r.t. p is 0
    // (since |x/sigma|^p * ln(|x/sigma|) -> 0 as x -> 0)
    return scalar_t(0);
  }

  // d/d(p) = forward_value * (-normalized^p * ln(normalized))
  // Note: -ln(forward_value) = normalized^p (since forward_value = exp(-normalized^p))
  // So: normalized^p = -ln(forward_value)
  // d/d(p) = forward_value * (-(-ln(forward_value))) * ln(normalized) / normalized^p * normalized^p
  //        = forward_value * ln(forward_value) * ln(normalized) / (-ln(forward_value)) * (-ln(forward_value))
  // Simpler: just compute directly
  scalar_t norm_p = std::pow(normalized, p);
  scalar_t log_norm = std::log(normalized);

  return grad_out * forward_value * (-norm_p) * log_norm;
}

// Gradient w.r.t. sigma parameter
// d/d(sigma) exp(-|x/sigma|^p) = exp(...) * (-p) * |x/sigma|^(p-1) * (-|x|/sigma^2)
//                               = forward_value * p * |x|^p / sigma^(p+1)
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t generalized_normal_backward_sigma(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t p,
  scalar_t sigma,
  bool periodic,
  scalar_t forward_value
) {
  if (n == 1 || sigma == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }

  scalar_t center = denom / scalar_t(2);
  scalar_t x = std::abs(scalar_t(i) - center);

  if (x == scalar_t(0)) {
    return scalar_t(0);  // At center, gradient w.r.t. sigma is 0
  }

  // d/d(sigma) = forward_value * p * |x|^p / sigma^(p+1)
  scalar_t x_p = std::pow(x, p);
  scalar_t sigma_p1 = std::pow(sigma, p + scalar_t(1));

  return grad_out * forward_value * p * x_p / sigma_p1;
}

}  // namespace torchscience::kernel::window_function
