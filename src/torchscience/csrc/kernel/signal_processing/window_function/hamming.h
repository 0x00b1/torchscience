#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Hamming window: w[k] = 0.54 - 0.46 * cos(2*pi*k / denom)
// Coefficients chosen to minimize first side lobe level
template<typename scalar_t>
inline scalar_t hamming(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return scalar_t(0.54) - scalar_t(0.46) * std::cos(x);
}

}  // namespace torchscience::kernel::window_function
