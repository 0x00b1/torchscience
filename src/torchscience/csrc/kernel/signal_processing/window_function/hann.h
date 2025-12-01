#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Hann window: w[k] = 0.5 * (1 - cos(2*pi*k / denom))
// where denom = n-1 (symmetric) or n (periodic)
template<typename scalar_t>
inline scalar_t hann(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return scalar_t(0.5) * (scalar_t(1) - std::cos(x));
}

}  // namespace torchscience::kernel::window_function
