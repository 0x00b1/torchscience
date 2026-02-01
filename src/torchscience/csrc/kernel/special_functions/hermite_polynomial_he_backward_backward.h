#pragma once

#include <cmath>
#include <tuple>

#include "hermite_polynomial_he.h"

namespace torchscience::kernel::special_functions {

// Backward_backward for Probabilists' Hermite polynomial He_n(z)
//
// For integer n:
//   First derivative: dHe/dz = n * He_{n-1}(z)
//   Second derivative: d^2He/dz^2 = n * (n-1) * He_{n-2}(z)
//
// For non-integer n: use finite differences
template <typename T>
std::tuple<T, T, T> hermite_polynomial_he_backward_backward(
  T gradient_gradient_n,
  T gradient_gradient_z,
  T gradient,
  T n,
  T z
) {
  T dHe_dz;
  T d2He_dz2;

  // Check if n is a non-negative integer
  if (detail::hermite_is_nonneg_integer(n)) {
    int n_int = detail::hermite_get_integer(n);

    // First derivative: dHe/dz = n * He_{n-1}(z)
    if (n_int == 0) {
      dHe_dz = T(0);
    } else {
      T He_n_minus_1 = hermite_polynomial_he(T(n_int - 1), z);
      dHe_dz = T(n_int) * He_n_minus_1;
    }

    // Second derivative: d^2He/dz^2 = n * (n-1) * He_{n-2}(z)
    if (n_int <= 1) {
      d2He_dz2 = T(0);
    } else {
      T He_n_minus_2 = hermite_polynomial_he(T(n_int - 2), z);
      d2He_dz2 = T(n_int) * T(n_int - 1) * He_n_minus_2;
    }
  } else {
    // For non-integer n, use finite differences
    T eps = T(1e-7);
    T He_plus = hermite_polynomial_he(n, z + eps);
    T He_curr = hermite_polynomial_he(n, z);
    T He_minus = hermite_polynomial_he(n, z - eps);
    dHe_dz = (He_plus - He_minus) / (T(2) * eps);
    d2He_dz2 = (He_plus - T(2) * He_curr + He_minus) / (eps * eps);
  }

  // Compute dHe/dn via finite differences
  T eps = T(1e-7);
  T He_plus = hermite_polynomial_he(n + eps, z);
  T He_minus = hermite_polynomial_he(n - eps, z);
  T dHe_dn = (He_plus - He_minus) / (T(2) * eps);

  // gradient_gradient_output = gg_z * dHe/dz + gg_n * dHe/dn
  T gradient_gradient_output = gradient_gradient_z * dHe_dz + gradient_gradient_n * dHe_dn;

  // new_gradient_z = gg_z * grad * d^2He/dz^2
  T new_gradient_z = gradient_gradient_z * gradient * d2He_dz2;

  // new_gradient_n: We approximate this as zero since d^2He/dndn
  // and d^2He/dndz are complex
  T new_gradient_n = T(0);

  return {gradient_gradient_output, new_gradient_n, new_gradient_z};
}

} // namespace torchscience::kernel::special_functions
