#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Flat-top window (5-term cosine window)
// Used for accurate amplitude measurements in spectral analysis.
// Has a very flat passband but can have negative values.
// Coefficients match scipy.signal.windows.flattop (D'Antona & Ferrero, 2006)
//
// w[k] = a0 - a1*cos(2*pi*k/denom) + a2*cos(4*pi*k/denom)
//        - a3*cos(6*pi*k/denom) + a4*cos(8*pi*k/denom)
template<typename scalar_t>
inline scalar_t flat_top(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  // Flat-top coefficients (same as scipy.signal.windows.flattop)
  scalar_t a0 = scalar_t(0.21557895);
  scalar_t a1 = scalar_t(0.41663158);
  scalar_t a2 = scalar_t(0.277263158);
  scalar_t a3 = scalar_t(0.083578947);
  scalar_t a4 = scalar_t(0.006947368);

  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return a0 - a1 * std::cos(x) + a2 * std::cos(scalar_t(2) * x)
       - a3 * std::cos(scalar_t(3) * x) + a4 * std::cos(scalar_t(4) * x);
}

}  // namespace torchscience::kernel::window_function
