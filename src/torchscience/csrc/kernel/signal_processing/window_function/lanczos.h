#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Lanczos window (sinc window): w[k] = sinc(2k/denom - 1)
// where sinc(x) = sin(pi*x)/(pi*x) with sinc(0) = 1
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t lanczos(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = scalar_t(2) * scalar_t(i) / denom - scalar_t(1);
  if (std::abs(x) < scalar_t(1e-10)) {
    return scalar_t(1);
  }
  scalar_t pi_x = static_cast<scalar_t>(M_PI) * x;
  return std::sin(pi_x) / pi_x;
}

}  // namespace torchscience::kernel::window_function
