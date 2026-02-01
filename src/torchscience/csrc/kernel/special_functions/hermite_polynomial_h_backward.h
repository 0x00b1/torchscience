#pragma once

#include <cmath>
#include <tuple>

#include "hermite_polynomial_h.h"

namespace torchscience::kernel::special_functions {

// Backward for Physicists' Hermite polynomial H_n(z)
//
// For integer n: dH/dz = 2n * H_{n-1}(z) (exact)
// For non-integer n: use finite differences (approximate)
//
// dH/dn: Always computed via finite differences
template <typename T>
std::tuple<T, T> hermite_polynomial_h_backward(T gradient, T n, T z) {
  T gradient_z;

  // Check if n is a non-negative integer
  if (detail::hermite_is_nonneg_integer(n)) {
    // For non-negative integer n, use the exact formula: dH/dz = 2n * H_{n-1}(z)
    int n_int = detail::hermite_get_integer(n);
    if (n_int == 0) {
      // dH_0/dz = 0
      gradient_z = T(0);
    } else {
      T H_n_minus_1 = detail::hermite_integer_recurrence(n_int - 1, z);
      T dH_dz = T(2) * T(n_int) * H_n_minus_1;
      gradient_z = gradient * dH_dz;
    }
  } else {
    // For non-integer n, use finite differences
    T eps = T(1e-7);
    T H_plus = hermite_polynomial_h(n, z + eps);
    T H_minus = hermite_polynomial_h(n, z - eps);
    T dH_dz = (H_plus - H_minus) / (T(2) * eps);
    gradient_z = gradient * dH_dz;
  }

  // dH/dn via finite differences
  T eps = T(1e-7);
  T H_plus = hermite_polynomial_h(n + eps, z);
  T H_minus = hermite_polynomial_h(n - eps, z);
  T dH_dn = (H_plus - H_minus) / (T(2) * eps);
  T gradient_n = gradient * dH_dn;

  return {gradient_n, gradient_z};
}

} // namespace torchscience::kernel::special_functions
