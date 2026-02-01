#pragma once

#include <cmath>
#include <tuple>

#include "hermite_polynomial_he.h"

namespace torchscience::kernel::special_functions {

// Backward for Probabilists' Hermite polynomial He_n(z)
//
// Using the chain rule from He_n(z) = 2^(-n/2) * H_n(z/sqrt(2)):
//
// dHe/dz = 2^(-n/2) * (1/sqrt(2)) * dH/dz(z/sqrt(2))
//        = 2^(-n/2) * (1/sqrt(2)) * 2n * H_{n-1}(z/sqrt(2))
//        = 2^(-(n+1)/2) * 2n * H_{n-1}(z/sqrt(2))
//        = n * 2^(-(n-1)/2) * H_{n-1}(z/sqrt(2))
//        = n * He_{n-1}(z)
//
// For integer n: dHe/dz = n * He_{n-1}(z)
// For non-integer n: use finite differences
//
// dHe/dn: use finite differences
template <typename T>
std::tuple<T, T> hermite_polynomial_he_backward(T gradient, T n, T z) {
  T gradient_z;

  // Check if n is a non-negative integer
  if (detail::hermite_is_nonneg_integer(n)) {
    int n_int = detail::hermite_get_integer(n);
    if (n_int == 0) {
      // dHe_0/dz = 0
      gradient_z = T(0);
    } else {
      // dHe_n/dz = n * He_{n-1}(z)
      T He_n_minus_1 = hermite_polynomial_he(T(n_int - 1), z);
      T dHe_dz = T(n_int) * He_n_minus_1;
      gradient_z = gradient * dHe_dz;
    }
  } else {
    // For non-integer n, use finite differences
    T eps = T(1e-7);
    T He_plus = hermite_polynomial_he(n, z + eps);
    T He_minus = hermite_polynomial_he(n, z - eps);
    T dHe_dz = (He_plus - He_minus) / (T(2) * eps);
    gradient_z = gradient * dHe_dz;
  }

  // dHe/dn via finite differences
  T eps = T(1e-7);
  T He_plus = hermite_polynomial_he(n + eps, z);
  T He_minus = hermite_polynomial_he(n - eps, z);
  T dHe_dn = (He_plus - He_minus) / (T(2) * eps);
  T gradient_n = gradient * dHe_dn;

  return {gradient_n, gradient_z};
}

} // namespace torchscience::kernel::special_functions
