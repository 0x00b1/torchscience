#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience::kernel::window_function {

// Sine window (also called cosine window in some literature)
// w[k] = sin(pi * k / denom)
// This is mathematically identical to the cosine window - scipy calls it "cosine"
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t sine(int64_t i, int64_t n, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }
  scalar_t x = static_cast<scalar_t>(M_PI) * scalar_t(i) / denom;
  return std::sin(x);
}

}  // namespace torchscience::kernel::window_function
