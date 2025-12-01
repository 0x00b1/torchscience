#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Cosine (sine) window: w[k] = sin(pi * k / denom)
// Also known as the sine window or half-cosine window
template<typename scalar_t>
inline scalar_t cosine(int64_t i, int64_t n, bool periodic) {
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
