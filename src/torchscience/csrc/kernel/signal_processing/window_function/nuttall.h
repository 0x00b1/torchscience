#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Nuttall window (4-term cosine):
// w[k] = a0 - a1*cos(2*pi*k/denom) + a2*cos(4*pi*k/denom) - a3*cos(6*pi*k/denom)
// where a0=0.355768, a1=0.487396, a2=0.144232, a3=0.012604
// Minimum 4-term Blackman-Harris window with continuous first derivative
template<typename scalar_t>
inline scalar_t nuttall(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  scalar_t a0 = scalar_t(0.355768);
  scalar_t a1 = scalar_t(0.487396);
  scalar_t a2 = scalar_t(0.144232);
  scalar_t a3 = scalar_t(0.012604);

  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return a0 - a1 * std::cos(x) + a2 * std::cos(scalar_t(2) * x) - a3 * std::cos(scalar_t(3) * x);
}

}  // namespace torchscience::kernel::window_function
