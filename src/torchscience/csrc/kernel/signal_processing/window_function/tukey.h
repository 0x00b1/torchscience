#pragma once

#include <cmath>
#include "common.h"

namespace torchscience::kernel::window_function {

// Tukey (tapered cosine) window
// alpha = 0: rectangular window
// alpha = 1: Hann window
// alpha in (0, 1): tapered cosine
template<typename scalar_t>
inline scalar_t tukey(int64_t i, int64_t n, scalar_t alpha, bool periodic) {
  if (n == 1) {
    return scalar_t(1);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(1);
  }

  // Clamp alpha to [0, 1]
  if (alpha <= scalar_t(0)) {
    return scalar_t(1);  // rectangular
  }
  if (alpha >= scalar_t(1)) {
    // Hann window
    scalar_t x = scalar_t(2) * M_PI * scalar_t(i) / denom;
    return scalar_t(0.5) * (scalar_t(1) - std::cos(x));
  }

  // Tapered cosine regions
  scalar_t width = alpha * denom / scalar_t(2);
  scalar_t x = scalar_t(i);

  if (x < width) {
    // Left taper region
    return scalar_t(0.5) * (scalar_t(1) - std::cos(M_PI * x / width));
  } else if (x <= denom - width) {
    // Flat region
    return scalar_t(1);
  } else {
    // Right taper region
    return scalar_t(0.5) * (scalar_t(1) - std::cos(M_PI * (denom - x) / width));
  }
}

// Gradient w.r.t. alpha parameter
template<typename scalar_t>
inline scalar_t tukey_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t alpha,
  bool periodic,
  scalar_t /* forward_value */
) {
  if (n == 1) {
    return scalar_t(0);
  }
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  if (denom == scalar_t(0)) {
    return scalar_t(0);
  }

  // For alpha <= 0 or alpha >= 1, gradient is 0
  // (boundary cases where changing alpha doesn't affect output)
  if (alpha <= scalar_t(0) || alpha >= scalar_t(1)) {
    return scalar_t(0);
  }

  scalar_t width = alpha * denom / scalar_t(2);
  scalar_t x = scalar_t(i);

  if (x < width) {
    // Left taper region
    // w = 0.5 * (1 - cos(pi * x / width))
    // d/d(alpha) = 0.5 * sin(pi * x / width) * (-pi * x / width^2) * (denom / 2)
    //            = 0.5 * sin(pi * x / width) * (-pi * x * denom / 2) / width^2
    scalar_t arg = M_PI * x / width;
    scalar_t d_width_d_alpha = denom / scalar_t(2);
    scalar_t d_arg_d_width = -M_PI * x / (width * width);
    return grad_out * scalar_t(0.5) * std::sin(arg) * d_arg_d_width * d_width_d_alpha;
  } else if (x <= denom - width) {
    // Flat region - output is constant 1, gradient is 0
    return scalar_t(0);
  } else {
    // Right taper region
    // w = 0.5 * (1 - cos(pi * (denom - x) / width))
    scalar_t dist = denom - x;
    scalar_t arg = M_PI * dist / width;
    scalar_t d_width_d_alpha = denom / scalar_t(2);
    scalar_t d_arg_d_width = -M_PI * dist / (width * width);
    return grad_out * scalar_t(0.5) * std::sin(arg) * d_arg_d_width * d_width_d_alpha;
  }
}

}  // namespace torchscience::kernel::window_function
