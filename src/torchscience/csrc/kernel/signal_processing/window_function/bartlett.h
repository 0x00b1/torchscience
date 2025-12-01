#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Bartlett (triangular) window: w[k] = 1 - |k - (n-1)/2| / ((n-1)/2)
// Also known as the triangular window
template<typename scalar_t>
inline scalar_t bartlett(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t half = denom / scalar_t(2);
  scalar_t val = scalar_t(1) - std::abs(scalar_t(i) - half) / half;
  return val;
}

}  // namespace torchscience::kernel::window_function
