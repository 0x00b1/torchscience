#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Blackman-Harris 4-term window (minimum 4-term)
// w[k] = a0 - a1*cos(2*pi*k/denom) + a2*cos(4*pi*k/denom) - a3*cos(6*pi*k/denom)
// Coefficients: a0=0.35875, a1=0.48829, a2=0.14128, a3=0.01168
// These coefficients match scipy.signal.windows.blackmanharris
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t blackman_harris(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  // Standard Blackman-Harris coefficients (same as scipy)
  scalar_t a0 = scalar_t(0.35875);
  scalar_t a1 = scalar_t(0.48829);
  scalar_t a2 = scalar_t(0.14128);
  scalar_t a3 = scalar_t(0.01168);

  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return a0 - a1 * std::cos(x) + a2 * std::cos(scalar_t(2) * x) - a3 * std::cos(scalar_t(3) * x);
}

}  // namespace torchscience::kernel::window_function
