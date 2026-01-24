#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// General Hamming window: w[k] = alpha - (1 - alpha) * cos(2*pi*k / denom)
// Standard Hamming uses alpha = 0.54, Hann uses alpha = 0.5
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t general_hamming(int64_t i, int64_t n, scalar_t alpha, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return alpha - (scalar_t(1) - alpha) * std::cos(x);
}

// Gradient w.r.t. alpha: d/d(alpha) = 1 + cos(x)
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t general_hamming_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t alpha,
  bool periodic,
  scalar_t forward_value
) {
  (void)alpha;
  (void)forward_value;
  if (n == 1) {
    return scalar_t(0);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return grad_out * (scalar_t(1) + std::cos(x));
}

}  // namespace torchscience::kernel::window_function
