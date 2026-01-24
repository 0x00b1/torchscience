#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Welch window: w[k] = 1 - ((k - center) / center)^2
// Parabolic shape, also known as Riesz or Parzen (confusingly)
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t welch(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t center = denom / scalar_t(2);
  scalar_t x = (scalar_t(i) - center) / center;
  return scalar_t(1) - x * x;
}

}  // namespace torchscience::kernel::window_function
