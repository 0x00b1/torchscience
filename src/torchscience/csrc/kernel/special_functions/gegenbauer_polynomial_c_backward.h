#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "gegenbauer_polynomial_c.h"

namespace torchscience::kernel::special_functions {

// Backward for Gegenbauer polynomial C_n^lambda(z)
//
// Derivative with respect to z:
// dC_n^lambda/dz = 2*lambda * C_{n-1}^{lambda+1}(z)
//
// Derivatives with respect to n and lambda are computed via finite differences
// since analytical forms are complex.
template <typename T>
std::tuple<T, T, T> gegenbauer_polynomial_c_backward(T gradient, T n, T lambda, T z) {
  // dC/dz = 2*lambda * C_{n-1}^{lambda+1}(z)
  T gradient_z;
  if (std::abs(n) < T(1e-10)) {
    // C_0^lambda(z) = 1, derivative is 0
    gradient_z = T(0);
  } else {
    T C_deriv = gegenbauer_polynomial_c(n - T(1), lambda + T(1), z);
    gradient_z = gradient * T(2) * lambda * C_deriv;
  }

  // dC/dn via finite differences
  T eps = T(1e-7);
  T C_plus_n = gegenbauer_polynomial_c(n + eps, lambda, z);
  T C_minus_n = gegenbauer_polynomial_c(n - eps, lambda, z);
  T gradient_n = gradient * (C_plus_n - C_minus_n) / (T(2) * eps);

  // dC/dlambda via finite differences
  T C_plus_lambda = gegenbauer_polynomial_c(n, lambda + eps, z);
  T C_minus_lambda = gegenbauer_polynomial_c(n, lambda - eps, z);
  T gradient_lambda = gradient * (C_plus_lambda - C_minus_lambda) / (T(2) * eps);

  return {gradient_n, gradient_lambda, gradient_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> gegenbauer_polynomial_c_backward(
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> lambda, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> zero(T(0), T(0));

  // dC/dz = 2*lambda * C_{n-1}^{lambda+1}(z)
  c10::complex<T> gradient_z;
  if (std::abs(n) < T(1e-10)) {
    gradient_z = zero;
  } else {
    c10::complex<T> C_deriv = gegenbauer_polynomial_c(n - one, lambda + one, z);
    gradient_z = gradient * two * lambda * C_deriv;
  }

  // dC/dn via finite differences
  c10::complex<T> eps(T(1e-7), T(0));
  c10::complex<T> C_plus_n = gegenbauer_polynomial_c(n + eps, lambda, z);
  c10::complex<T> C_minus_n = gegenbauer_polynomial_c(n - eps, lambda, z);
  c10::complex<T> gradient_n = gradient * (C_plus_n - C_minus_n) / (two * eps);

  // dC/dlambda via finite differences
  c10::complex<T> C_plus_lambda = gegenbauer_polynomial_c(n, lambda + eps, z);
  c10::complex<T> C_minus_lambda = gegenbauer_polynomial_c(n, lambda - eps, z);
  c10::complex<T> gradient_lambda = gradient * (C_plus_lambda - C_minus_lambda) / (two * eps);

  return {gradient_n, gradient_lambda, gradient_z};
}

} // namespace torchscience::kernel::special_functions
