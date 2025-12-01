#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Bartlett-Hann window
// Combination of Bartlett and Hann windows
// w[k] = a0 - a1*|k/denom - 0.5| - a2*cos(2*pi*k/denom)
// Coefficients: a0=0.62, a1=0.48, a2=0.38
// These coefficients match scipy.signal.windows.barthann
template<typename scalar_t>
inline scalar_t bartlett_hann(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  // Bartlett-Hann coefficients (same as scipy)
  scalar_t a0 = scalar_t(0.62);
  scalar_t a1 = scalar_t(0.48);
  scalar_t a2 = scalar_t(0.38);

  scalar_t frac = scalar_t(i) / denom;
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * frac;
  return a0 - a1 * std::abs(frac - scalar_t(0.5)) - a2 * std::cos(x);
}

}  // namespace torchscience::kernel::window_function
