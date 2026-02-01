#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_polynomial_p.h"

namespace torchscience::kernel::special_functions {

// Backward for Jacobi polynomial P_n^(alpha,beta)(z)
//
// Derivative with respect to z:
// dP_n^(alpha,beta)/dz = (n + alpha + beta + 1)/2 * P_{n-1}^(alpha+1,beta+1)(z)
//
// Derivatives with respect to n, alpha, and beta are computed via finite differences
// since analytical forms are complex.
template <typename T>
std::tuple<T, T, T, T> jacobi_polynomial_p_backward(T gradient, T n, T alpha, T beta, T z) {
  // dP/dz = (n + alpha + beta + 1)/2 * P_{n-1}^(alpha+1, beta+1)(z)
  T gradient_z;
  if (std::abs(n) < T(1e-10)) {
    // P_0^(alpha,beta)(z) = 1, derivative is 0
    gradient_z = T(0);
  } else {
    T P_deriv = jacobi_polynomial_p(n - T(1), alpha + T(1), beta + T(1), z);
    gradient_z = gradient * (n + alpha + beta + T(1)) / T(2) * P_deriv;
  }

  // dP/dn via finite differences
  T eps = T(1e-7);
  T P_plus_n = jacobi_polynomial_p(n + eps, alpha, beta, z);
  T P_minus_n = jacobi_polynomial_p(n - eps, alpha, beta, z);
  T gradient_n = gradient * (P_plus_n - P_minus_n) / (T(2) * eps);

  // dP/dalpha via finite differences
  T P_plus_alpha = jacobi_polynomial_p(n, alpha + eps, beta, z);
  T P_minus_alpha = jacobi_polynomial_p(n, alpha - eps, beta, z);
  T gradient_alpha = gradient * (P_plus_alpha - P_minus_alpha) / (T(2) * eps);

  // dP/dbeta via finite differences
  T P_plus_beta = jacobi_polynomial_p(n, alpha, beta + eps, z);
  T P_minus_beta = jacobi_polynomial_p(n, alpha, beta - eps, z);
  T gradient_beta = gradient * (P_plus_beta - P_minus_beta) / (T(2) * eps);

  return {gradient_n, gradient_alpha, gradient_beta, gradient_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_polynomial_p_backward(
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> alpha, c10::complex<T> beta, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> zero(T(0), T(0));

  // dP/dz = (n + alpha + beta + 1)/2 * P_{n-1}^(alpha+1, beta+1)(z)
  c10::complex<T> gradient_z;
  if (std::abs(n) < T(1e-10)) {
    gradient_z = zero;
  } else {
    c10::complex<T> P_deriv = jacobi_polynomial_p(n - one, alpha + one, beta + one, z);
    gradient_z = gradient * (n + alpha + beta + one) / two * P_deriv;
  }

  // dP/dn via finite differences
  c10::complex<T> eps(T(1e-7), T(0));
  c10::complex<T> P_plus_n = jacobi_polynomial_p(n + eps, alpha, beta, z);
  c10::complex<T> P_minus_n = jacobi_polynomial_p(n - eps, alpha, beta, z);
  c10::complex<T> gradient_n = gradient * (P_plus_n - P_minus_n) / (two * eps);

  // dP/dalpha via finite differences
  c10::complex<T> P_plus_alpha = jacobi_polynomial_p(n, alpha + eps, beta, z);
  c10::complex<T> P_minus_alpha = jacobi_polynomial_p(n, alpha - eps, beta, z);
  c10::complex<T> gradient_alpha = gradient * (P_plus_alpha - P_minus_alpha) / (two * eps);

  // dP/dbeta via finite differences
  c10::complex<T> P_plus_beta = jacobi_polynomial_p(n, alpha, beta + eps, z);
  c10::complex<T> P_minus_beta = jacobi_polynomial_p(n, alpha, beta - eps, z);
  c10::complex<T> gradient_beta = gradient * (P_plus_beta - P_minus_beta) / (two * eps);

  return {gradient_n, gradient_alpha, gradient_beta, gradient_z};
}

} // namespace torchscience::kernel::special_functions
