#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Blackman window: w[k] = 0.42 - 0.5*cos(2*pi*k/denom) + 0.08*cos(4*pi*k/denom)
// 3-term cosine window with good side lobe suppression
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t blackman(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = scalar_t(2) * static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return scalar_t(0.42) - scalar_t(0.5) * std::cos(x) + scalar_t(0.08) * std::cos(scalar_t(2) * x);
}

}  // namespace torchscience::kernel::window_function
