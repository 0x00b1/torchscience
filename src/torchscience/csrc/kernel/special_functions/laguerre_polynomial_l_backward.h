#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "confluent_hypergeometric_m.h"
#include "gamma.h"
#include "digamma.h"
#include "laguerre_polynomial_l.h"

namespace torchscience::kernel::special_functions {

// Backward pass for generalized Laguerre polynomial L_n^alpha(z)
//
// L_n^alpha(z) = C(n+alpha, n) * 1F1(-n; alpha+1; z)
// where C(n+alpha, n) = Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1))
//
// Gradients:
// dL/dz = C * d(1F1)/dz = C * (-n/(alpha+1)) * 1F1(-n+1; alpha+2; z)
//       = -n/(alpha+1) * C * 1F1(-n+1; alpha+2; z)
//
// dL/dn and dL/dalpha use finite differences for robustness

template <typename T>
std::tuple<T, T, T> laguerre_polynomial_l_backward(T gradient, T n, T alpha, T z) {
  T eps = T(1e-7);

  // Gradient with respect to z
  // d/dz 1F1(a; b; z) = (a/b) * 1F1(a+1; b+1; z)
  // Here a = -n, b = alpha+1
  // d/dz 1F1(-n; alpha+1; z) = (-n/(alpha+1)) * 1F1(-n+1; alpha+2; z)
  //
  // dL/dz = coeff * (-n/(alpha+1)) * 1F1(-n+1; alpha+2; z)

  T grad_z;

  // For n = 0, L_0^alpha(z) = 1, so dL/dz = 0
  if (std::abs(n) < eps) {
    grad_z = T(0);
  } else {
    // Compute coefficient
    T log_coeff = log_gamma(n + alpha + T(1)) - log_gamma(alpha + T(1)) - log_gamma(n + T(1));
    T coeff = std::exp(log_coeff);

    // Compute derivative of 1F1
    T a = -n;
    T b = alpha + T(1);
    T dM_dz = (a / b) * confluent_hypergeometric_m(a + T(1), b + T(1), z);

    grad_z = coeff * dM_dz;
  }

  // Gradient with respect to n using finite differences
  T L_plus = laguerre_polynomial_l(n + eps, alpha, z);
  T L_minus = laguerre_polynomial_l(n - eps, alpha, z);
  T grad_n = (L_plus - L_minus) / (T(2) * eps);

  // Gradient with respect to alpha using finite differences
  L_plus = laguerre_polynomial_l(n, alpha + eps, z);
  L_minus = laguerre_polynomial_l(n, alpha - eps, z);
  T grad_alpha = (L_plus - L_minus) / (T(2) * eps);

  return {gradient * grad_n, gradient * grad_alpha, gradient * grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
laguerre_polynomial_l_backward(c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> alpha, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> zero(T(0), T(0));
  T eps_val = T(1e-7);
  c10::complex<T> eps(eps_val, T(0));

  // Gradient with respect to z
  c10::complex<T> grad_z;

  if (std::abs(n) < eps_val) {
    grad_z = zero;
  } else {
    c10::complex<T> log_coeff = log_gamma(n + alpha + one) - log_gamma(alpha + one) - log_gamma(n + one);
    c10::complex<T> coeff = std::exp(log_coeff);

    c10::complex<T> a = -n;
    c10::complex<T> b = alpha + one;
    c10::complex<T> dM_dz = (a / b) * confluent_hypergeometric_m(a + one, b + one, z);

    grad_z = coeff * dM_dz;
  }

  // Gradient with respect to n using finite differences
  c10::complex<T> L_plus = laguerre_polynomial_l(n + eps, alpha, z);
  c10::complex<T> L_minus = laguerre_polynomial_l(n - eps, alpha, z);
  c10::complex<T> grad_n = (L_plus - L_minus) / (c10::complex<T>(T(2), T(0)) * eps);

  // Gradient with respect to alpha using finite differences
  L_plus = laguerre_polynomial_l(n, alpha + eps, z);
  L_minus = laguerre_polynomial_l(n, alpha - eps, z);
  c10::complex<T> grad_alpha = (L_plus - L_minus) / (c10::complex<T>(T(2), T(0)) * eps);

  // For complex holomorphic functions, use Wirtinger derivative convention
  return {
    gradient * std::conj(grad_n),
    gradient * std::conj(grad_alpha),
    gradient * std::conj(grad_z)
  };
}

} // namespace torchscience::kernel::special_functions
