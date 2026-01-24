#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace torchscience::kernel::window_function {

// Hann-Poisson window: w[k] = hann[k] * poisson[k]
// where hann[k] = 0.5 * (1 - cos(2 * pi * k / denom))
// and poisson[k] = exp(-alpha * |denom - 2k| / denom)
// For symmetric: denom = n-1, for periodic: denom = n
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t hann_poisson(int64_t i, int64_t n, scalar_t alpha, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  // Hann component: 0.5 * (1 - cos(2 * pi * k / denom))
  scalar_t phase = scalar_t(2) * scalar_t(M_PI) * scalar_t(i) / denom;
  scalar_t hann = scalar_t(0.5) * (scalar_t(1) - std::cos(phase));

  // Poisson component: exp(-alpha * |denom - 2k| / denom)
  scalar_t abs_diff = std::abs(denom - scalar_t(2) * scalar_t(i));
  scalar_t poisson = std::exp(-alpha * abs_diff / denom);

  return hann * poisson;
}

// Gradient w.r.t. alpha parameter
// Let f = hann * poisson, where poisson = exp(-alpha * x), x = |denom - 2k| / denom
// d/d(alpha) = hann * poisson * (-x) = -forward_value * x
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t hann_poisson_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t alpha,
  bool periodic,
  scalar_t forward_value
) {
  (void)alpha;  // alpha not needed for gradient computation

  if (n == 1) {
    return scalar_t(0);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }

  // x = |denom - 2k| / denom
  scalar_t abs_diff = std::abs(denom - scalar_t(2) * scalar_t(i));
  scalar_t x = abs_diff / denom;

  // d/d(alpha) = -forward_value * x
  return grad_out * (-forward_value) * x;
}

}  // namespace torchscience::kernel::window_function
