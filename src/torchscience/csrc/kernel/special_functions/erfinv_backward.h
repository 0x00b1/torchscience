#pragma once

#include <cmath>

#include "erfinv.h"

namespace torchscience::kernel::special_functions {

// Backward for erfinv: d/dx erfinv(x) = sqrt(pi)/2 * exp(erfinv(x)^2)
//
// Derivation:
//   Let y = erfinv(x), so erf(y) = x
//   Differentiating both sides: erf'(y) * y' = 1
//   erf'(y) = 2/sqrt(pi) * exp(-y^2)
//   So y' = sqrt(pi)/2 * exp(y^2)
//
// This derivative is always positive and grows rapidly as |x| -> 1
template <typename T>
T erfinv_backward(T gradient, T x) {
  const T sqrt_pi_over_2 = static_cast<T>(0.8862269254527580136490837416705725914);

  T y = erfinv(x);
  T y2 = y * y;

  // Handle edge cases where y is infinite
  if (std::isinf(y)) {
    // At x = +/-1, the derivative is infinite
    return gradient * std::numeric_limits<T>::infinity();
  }

  T deriv = sqrt_pi_over_2 * std::exp(y2);
  return gradient * deriv;
}

}  // namespace torchscience::kernel::special_functions
