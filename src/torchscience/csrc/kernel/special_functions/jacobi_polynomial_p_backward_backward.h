#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_polynomial_p.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Jacobi polynomial P_n^(alpha,beta)(z)
//
// d^2P/dz^2 = (n + alpha + beta + 1)(n + alpha + beta + 2)/4 * P_{n-2}^(alpha+2,beta+2)(z)
template <typename T>
std::tuple<T, T, T, T, T> jacobi_polynomial_p_backward_backward(
    T gradient_gradient_n,
    T gradient_gradient_alpha,
    T gradient_gradient_beta,
    T gradient_gradient_z,
    T gradient,
    T n,
    T alpha,
    T beta,
    T z
) {
  // Second derivative d^2P/dz^2
  T d2P_dz2;
  if (std::abs(n) < T(1e-10) || std::abs(n - T(1)) < T(1e-10)) {
    // P_0 and P_1 have zero second derivative in z
    d2P_dz2 = T(0);
  } else {
    T P_second = jacobi_polynomial_p(n - T(2), alpha + T(2), beta + T(2), z);
    T coeff = (n + alpha + beta + T(1)) * (n + alpha + beta + T(2)) / T(4);
    d2P_dz2 = coeff * P_second;
  }

  // First derivative dP/dz for gradient_gradient_output
  T dP_dz;
  if (std::abs(n) < T(1e-10)) {
    dP_dz = T(0);
  } else {
    T P_deriv = jacobi_polynomial_p(n - T(1), alpha + T(1), beta + T(1), z);
    dP_dz = (n + alpha + beta + T(1)) / T(2) * P_deriv;
  }

  // gradient_gradient_output computation
  T gradient_gradient_output = gradient_gradient_z * dP_dz;

  // Contribution from gradient_gradient_n, gradient_gradient_alpha, gradient_gradient_beta
  T eps = T(1e-7);
  if (std::abs(gradient_gradient_n) > T(1e-15)) {
    T P_plus_n = jacobi_polynomial_p(n + eps, alpha, beta, z);
    T P_minus_n = jacobi_polynomial_p(n - eps, alpha, beta, z);
    T dP_dn = (P_plus_n - P_minus_n) / (T(2) * eps);
    gradient_gradient_output += gradient_gradient_n * dP_dn;
  }

  if (std::abs(gradient_gradient_alpha) > T(1e-15)) {
    T P_plus_alpha = jacobi_polynomial_p(n, alpha + eps, beta, z);
    T P_minus_alpha = jacobi_polynomial_p(n, alpha - eps, beta, z);
    T dP_dalpha = (P_plus_alpha - P_minus_alpha) / (T(2) * eps);
    gradient_gradient_output += gradient_gradient_alpha * dP_dalpha;
  }

  if (std::abs(gradient_gradient_beta) > T(1e-15)) {
    T P_plus_beta = jacobi_polynomial_p(n, alpha, beta + eps, z);
    T P_minus_beta = jacobi_polynomial_p(n, alpha, beta - eps, z);
    T dP_dbeta = (P_plus_beta - P_minus_beta) / (T(2) * eps);
    gradient_gradient_output += gradient_gradient_beta * dP_dbeta;
  }

  // Second-order gradient contributions
  T new_gradient_z = gradient_gradient_z * gradient * d2P_dz2;

  // Cross derivatives are computed via finite differences (simplified to zero)
  T new_gradient_n = T(0);
  T new_gradient_alpha = T(0);
  T new_gradient_beta = T(0);

  return {gradient_gradient_output, new_gradient_n, new_gradient_alpha, new_gradient_beta, new_gradient_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_polynomial_p_backward_backward(
    c10::complex<T> gradient_gradient_n,
    c10::complex<T> gradient_gradient_alpha,
    c10::complex<T> gradient_gradient_beta,
    c10::complex<T> gradient_gradient_z,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    c10::complex<T> z
) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> four(T(4), T(0));
  c10::complex<T> zero(T(0), T(0));

  // Second derivative d^2P/dz^2
  c10::complex<T> d2P_dz2;
  if (std::abs(n) < T(1e-10) || std::abs(n - one) < T(1e-10)) {
    d2P_dz2 = zero;
  } else {
    c10::complex<T> P_second = jacobi_polynomial_p(n - two, alpha + two, beta + two, z);
    c10::complex<T> coeff = (n + alpha + beta + one) * (n + alpha + beta + two) / four;
    d2P_dz2 = coeff * P_second;
  }

  // First derivative dP/dz
  c10::complex<T> dP_dz;
  if (std::abs(n) < T(1e-10)) {
    dP_dz = zero;
  } else {
    c10::complex<T> P_deriv = jacobi_polynomial_p(n - one, alpha + one, beta + one, z);
    dP_dz = (n + alpha + beta + one) / two * P_deriv;
  }

  c10::complex<T> gradient_gradient_output = gradient_gradient_z * dP_dz;

  // Contributions from other gradient_gradients
  c10::complex<T> eps(T(1e-7), T(0));
  if (std::abs(gradient_gradient_n) > T(1e-15)) {
    c10::complex<T> P_plus_n = jacobi_polynomial_p(n + eps, alpha, beta, z);
    c10::complex<T> P_minus_n = jacobi_polynomial_p(n - eps, alpha, beta, z);
    c10::complex<T> dP_dn = (P_plus_n - P_minus_n) / (two * eps);
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_n * dP_dn;
  }

  if (std::abs(gradient_gradient_alpha) > T(1e-15)) {
    c10::complex<T> P_plus_alpha = jacobi_polynomial_p(n, alpha + eps, beta, z);
    c10::complex<T> P_minus_alpha = jacobi_polynomial_p(n, alpha - eps, beta, z);
    c10::complex<T> dP_dalpha = (P_plus_alpha - P_minus_alpha) / (two * eps);
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_alpha * dP_dalpha;
  }

  if (std::abs(gradient_gradient_beta) > T(1e-15)) {
    c10::complex<T> P_plus_beta = jacobi_polynomial_p(n, alpha, beta + eps, z);
    c10::complex<T> P_minus_beta = jacobi_polynomial_p(n, alpha, beta - eps, z);
    c10::complex<T> dP_dbeta = (P_plus_beta - P_minus_beta) / (two * eps);
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_beta * dP_dbeta;
  }

  c10::complex<T> new_gradient_z = gradient_gradient_z * gradient * d2P_dz2;
  c10::complex<T> new_gradient_n = zero;
  c10::complex<T> new_gradient_alpha = zero;
  c10::complex<T> new_gradient_beta = zero;

  return {gradient_gradient_output, new_gradient_n, new_gradient_alpha, new_gradient_beta, new_gradient_z};
}

} // namespace torchscience::kernel::special_functions
