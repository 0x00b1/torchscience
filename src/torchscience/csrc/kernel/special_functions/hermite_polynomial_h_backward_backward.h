#pragma once

#include <cmath>
#include <tuple>

#include "hermite_polynomial_h.h"

namespace torchscience::kernel::special_functions {

// Backward_backward for Physicists' Hermite polynomial H_n(z)
//
// For integer n:
//   First derivative: dH/dz = 2n * H_{n-1}(z)
//   Second derivative: d^2H/dz^2 = 4n(n-1) * H_{n-2}(z)
//
// For non-integer n: use finite differences
template <typename T>
std::tuple<T, T, T> hermite_polynomial_h_backward_backward(
  T gradient_gradient_n,
  T gradient_gradient_z,
  T gradient,
  T n,
  T z
) {
  T dH_dz;
  T d2H_dz2;

  // Check if n is a non-negative integer
  if (detail::hermite_is_nonneg_integer(n)) {
    int n_int = detail::hermite_get_integer(n);

    // First derivative: dH/dz = 2n * H_{n-1}(z)
    if (n_int == 0) {
      dH_dz = T(0);
    } else {
      T H_n_minus_1 = detail::hermite_integer_recurrence(n_int - 1, z);
      dH_dz = T(2) * T(n_int) * H_n_minus_1;
    }

    // Second derivative: d^2H/dz^2 = 4n(n-1) * H_{n-2}(z)
    if (n_int <= 1) {
      d2H_dz2 = T(0);
    } else {
      T H_n_minus_2 = detail::hermite_integer_recurrence(n_int - 2, z);
      d2H_dz2 = T(4) * T(n_int) * T(n_int - 1) * H_n_minus_2;
    }
  } else {
    // For non-integer n, use finite differences
    T eps = T(1e-7);
    T H_plus = hermite_polynomial_h(n, z + eps);
    T H_curr = hermite_polynomial_h(n, z);
    T H_minus = hermite_polynomial_h(n, z - eps);
    dH_dz = (H_plus - H_minus) / (T(2) * eps);
    d2H_dz2 = (H_plus - T(2) * H_curr + H_minus) / (eps * eps);
  }

  // Compute dH/dn via finite differences
  T eps = T(1e-7);
  T H_plus = hermite_polynomial_h(n + eps, z);
  T H_minus = hermite_polynomial_h(n - eps, z);
  T dH_dn = (H_plus - H_minus) / (T(2) * eps);

  // gradient_gradient_output = gg_z * dH/dz + gg_n * dH/dn
  T gradient_gradient_output = gradient_gradient_z * dH_dz + gradient_gradient_n * dH_dn;

  // new_gradient_z = gg_z * grad * d^2H/dz^2
  T new_gradient_z = gradient_gradient_z * gradient * d2H_dz2;

  // new_gradient_n: We approximate this as zero since d^2H/dndn
  // and d^2H/dndz are complex
  T new_gradient_n = T(0);

  return {gradient_gradient_output, new_gradient_n, new_gradient_z};
}

} // namespace torchscience::kernel::special_functions
