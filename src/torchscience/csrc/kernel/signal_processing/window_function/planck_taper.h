#pragma once

#include <cmath>
#include <c10/macros/Macros.h>
#include "common.h"

namespace torchscience {
namespace kernel {
namespace window_function {

// Planck-taper window forward
// Piecewise function:
// t = 0 or t = 1: w = 0
// 0 < t < epsilon: w = 1 / (1 + exp(Z⁺)) where Z⁺ = epsilon * (1/t + 1/(t - epsilon))
// epsilon <= t <= 1 - epsilon: w = 1
// 1 - epsilon < t < 1: w = 1 / (1 + exp(Z⁻)) where Z⁻ = epsilon * (1/(1-t) + 1/(1-t-epsilon))
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t planck_taper(int64_t i, int64_t n, scalar_t epsilon, bool periodic) {
  // Handle epsilon = 0 case (rectangular window - all ones)
  if (epsilon <= scalar_t(0)) {
    return scalar_t(1);
  }

  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t t = scalar_t(i) / denom;

  // Boundary points are exactly 0 (only when epsilon > 0)
  if (t <= scalar_t(0) || (t >= scalar_t(1) && !periodic)) {
    return scalar_t(0);
  }

  // Left taper region: 0 < t < epsilon
  if (t > scalar_t(0) && t < epsilon) {
    scalar_t z_plus = epsilon * (scalar_t(1) / t + scalar_t(1) / (t - epsilon));
    // Clamp to avoid overflow in exp
    if (z_plus > scalar_t(80)) {
      return scalar_t(0);
    }
    return scalar_t(1) / (scalar_t(1) + std::exp(z_plus));
  }

  // Right taper region: 1 - epsilon < t < 1
  if (t > scalar_t(1) - epsilon && t < scalar_t(1)) {
    scalar_t one_minus_t = scalar_t(1) - t;
    scalar_t z_minus = epsilon * (scalar_t(1) / one_minus_t + scalar_t(1) / (one_minus_t - epsilon));
    // Clamp to avoid overflow in exp
    if (z_minus > scalar_t(80)) {
      return scalar_t(0);
    }
    return scalar_t(1) / (scalar_t(1) + std::exp(z_minus));
  }

  // Flat region: epsilon <= t <= 1 - epsilon
  return scalar_t(1);
}

// Planck-taper window backward (gradient w.r.t. epsilon)
// For left taper:
// w = 1 / (1 + exp(Z⁺))
// dw/d(epsilon) = -w² * exp(Z⁺) * dZ⁺/d(epsilon)
// dZ⁺/d(epsilon) = 1/t + 1/(t-epsilon) + epsilon/(t-epsilon)²
//
// For right taper:
// w = 1 / (1 + exp(Z⁻))
// dw/d(epsilon) = -w² * exp(Z⁻) * dZ⁻/d(epsilon)
// dZ⁻/d(epsilon) = 1/(1-t) + 1/(1-t-epsilon) + epsilon/(1-t-epsilon)²
template<typename scalar_t>
C10_HOST_DEVICE inline scalar_t planck_taper_backward(
  scalar_t grad_out,
  int64_t i,
  int64_t n,
  scalar_t epsilon,
  bool periodic,
  scalar_t forward_value
) {
  scalar_t denom = window_denominator<scalar_t>(n, periodic);
  scalar_t t = scalar_t(i) / denom;

  // Boundary points have no gradient
  if (t <= scalar_t(0) || (t >= scalar_t(1) && !periodic)) {
    return scalar_t(0);
  }

  // Handle epsilon = 0 case
  if (epsilon <= scalar_t(0)) {
    return scalar_t(0);
  }

  // Left taper region: 0 < t < epsilon
  if (t > scalar_t(0) && t < epsilon) {
    scalar_t t_minus_eps = t - epsilon;
    scalar_t z_plus = epsilon * (scalar_t(1) / t + scalar_t(1) / t_minus_eps);

    // Clamp for numerical stability
    if (z_plus > scalar_t(80)) {
      return scalar_t(0);
    }

    scalar_t exp_z = std::exp(z_plus);
    scalar_t w = forward_value;

    // dZ⁺/d(epsilon) = 1/t + 1/(t-epsilon) + epsilon/(t-epsilon)²
    scalar_t dz_deps = scalar_t(1) / t + scalar_t(1) / t_minus_eps
                       + epsilon / (t_minus_eps * t_minus_eps);

    // dw/d(epsilon) = -w² * exp(Z⁺) * dZ⁺/d(epsilon)
    scalar_t dw_deps = -w * w * exp_z * dz_deps;

    return grad_out * dw_deps;
  }

  // Right taper region: 1 - epsilon < t < 1
  if (t > scalar_t(1) - epsilon && t < scalar_t(1)) {
    scalar_t one_minus_t = scalar_t(1) - t;
    scalar_t one_minus_t_minus_eps = one_minus_t - epsilon;
    scalar_t z_minus = epsilon * (scalar_t(1) / one_minus_t + scalar_t(1) / one_minus_t_minus_eps);

    // Clamp for numerical stability
    if (z_minus > scalar_t(80)) {
      return scalar_t(0);
    }

    scalar_t exp_z = std::exp(z_minus);
    scalar_t w = forward_value;

    // dZ⁻/d(epsilon) = 1/(1-t) + 1/(1-t-epsilon) + epsilon/(1-t-epsilon)²
    scalar_t dz_deps = scalar_t(1) / one_minus_t + scalar_t(1) / one_minus_t_minus_eps
                       + epsilon / (one_minus_t_minus_eps * one_minus_t_minus_eps);

    // dw/d(epsilon) = -w² * exp(Z⁻) * dZ⁻/d(epsilon)
    scalar_t dw_deps = -w * w * exp_z * dz_deps;

    return grad_out * dw_deps;
  }

  // Flat region: epsilon <= t <= 1 - epsilon
  // However, the boundary between flat and taper regions depends on epsilon
  // The gradient from this region change is complex to compute correctly
  // For simplicity, return 0 for the flat region
  return scalar_t(0);
}

}  // namespace window_function
}  // namespace kernel
}  // namespace torchscience
