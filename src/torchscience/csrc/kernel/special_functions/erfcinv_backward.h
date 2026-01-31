#pragma once

#include <cmath>

#include "erfcinv.h"

namespace torchscience::kernel::special_functions {

// Backward for erfcinv: d/dx erfcinv(x) = -sqrt(pi)/2 * exp(erfcinv(x)^2)
//
// Derivation:
//   Let y = erfcinv(x), so erfc(y) = x
//   Differentiating both sides: erfc'(y) * y' = 1
//   erfc'(y) = -2/sqrt(pi) * exp(-y^2)
//   So y' = -sqrt(pi)/2 * exp(y^2)
//
// This derivative is always negative (erfcinv is monotonically decreasing)
// and its magnitude grows rapidly as x -> 0 or x -> 2
template <typename T>
T erfcinv_backward(T gradient, T x) {
  const T neg_sqrt_pi_over_2 = static_cast<T>(-0.8862269254527580136490837416705725914);

  T y = erfcinv(x);
  T y2 = y * y;

  // Handle edge cases where y is infinite
  if (std::isinf(y)) {
    // At x = 0 or x = 2, the derivative is +/- infinite
    if (y > static_cast<T>(0)) {
      return gradient * (-std::numeric_limits<T>::infinity());
    } else {
      return gradient * std::numeric_limits<T>::infinity();
    }
  }

  T deriv = neg_sqrt_pi_over_2 * std::exp(y2);
  return gradient * deriv;
}

}  // namespace torchscience::kernel::special_functions
