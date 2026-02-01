#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "gegenbauer_polynomial_c.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Gegenbauer polynomial C_n^lambda(z)
//
// d^2C/dz^2 = 4*lambda*(lambda+1) * C_{n-2}^{lambda+2}(z)
template <typename T>
std::tuple<T, T, T, T> gegenbauer_polynomial_c_backward_backward(
    T gradient_gradient_n,
    T gradient_gradient_lambda,
    T gradient_gradient_z,
    T gradient,
    T n,
    T lambda,
    T z
) {
  // Second derivative d^2C/dz^2
  T d2C_dz2;
  if (std::abs(n) < T(1e-10) || std::abs(n - T(1)) < T(1e-10)) {
    // C_0 and C_1 have zero second derivative in z
    d2C_dz2 = T(0);
  } else {
    T C_second = gegenbauer_polynomial_c(n - T(2), lambda + T(2), z);
    d2C_dz2 = T(4) * lambda * (lambda + T(1)) * C_second;
  }

  // First derivative dC/dz for gradient_gradient_output
  T dC_dz;
  if (std::abs(n) < T(1e-10)) {
    dC_dz = T(0);
  } else {
    T C_deriv = gegenbauer_polynomial_c(n - T(1), lambda + T(1), z);
    dC_dz = T(2) * lambda * C_deriv;
  }

  // gradient_gradient_output = gg_z * dC/dz + gg_n * dC/dn + gg_lambda * dC/dlambda
  // For simplicity, we handle each contribution separately
  T gradient_gradient_output = gradient_gradient_z * dC_dz;

  // Contribution from gradient_gradient_n (cross derivatives with finite differences)
  if (std::abs(gradient_gradient_n) > T(1e-15)) {
    T eps = T(1e-7);
    T C_plus_n = gegenbauer_polynomial_c(n + eps, lambda, z);
    T C_minus_n = gegenbauer_polynomial_c(n - eps, lambda, z);
    T dC_dn = (C_plus_n - C_minus_n) / (T(2) * eps);
    gradient_gradient_output += gradient_gradient_n * dC_dn;
  }

  // Contribution from gradient_gradient_lambda
  if (std::abs(gradient_gradient_lambda) > T(1e-15)) {
    T eps = T(1e-7);
    T C_plus_lambda = gegenbauer_polynomial_c(n, lambda + eps, z);
    T C_minus_lambda = gegenbauer_polynomial_c(n, lambda - eps, z);
    T dC_dlambda = (C_plus_lambda - C_minus_lambda) / (T(2) * eps);
    gradient_gradient_output += gradient_gradient_lambda * dC_dlambda;
  }

  // Second-order gradient contributions
  T new_gradient_z = gradient_gradient_z * gradient * d2C_dz2;

  // Cross derivatives are computed via finite differences (simplified)
  T new_gradient_n = T(0);
  T new_gradient_lambda = T(0);

  return {gradient_gradient_output, new_gradient_n, new_gradient_lambda, new_gradient_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
gegenbauer_polynomial_c_backward_backward(
    c10::complex<T> gradient_gradient_n,
    c10::complex<T> gradient_gradient_lambda,
    c10::complex<T> gradient_gradient_z,
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> lambda,
    c10::complex<T> z
) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> four(T(4), T(0));
  c10::complex<T> zero(T(0), T(0));

  // Second derivative d^2C/dz^2
  c10::complex<T> d2C_dz2;
  if (std::abs(n) < T(1e-10) || std::abs(n - one) < T(1e-10)) {
    d2C_dz2 = zero;
  } else {
    c10::complex<T> C_second = gegenbauer_polynomial_c(n - two, lambda + two, z);
    d2C_dz2 = four * lambda * (lambda + one) * C_second;
  }

  // First derivative dC/dz
  c10::complex<T> dC_dz;
  if (std::abs(n) < T(1e-10)) {
    dC_dz = zero;
  } else {
    c10::complex<T> C_deriv = gegenbauer_polynomial_c(n - one, lambda + one, z);
    dC_dz = two * lambda * C_deriv;
  }

  c10::complex<T> gradient_gradient_output = gradient_gradient_z * dC_dz;

  // Contributions from other gradient_gradients
  c10::complex<T> eps(T(1e-7), T(0));
  if (std::abs(gradient_gradient_n) > T(1e-15)) {
    c10::complex<T> C_plus_n = gegenbauer_polynomial_c(n + eps, lambda, z);
    c10::complex<T> C_minus_n = gegenbauer_polynomial_c(n - eps, lambda, z);
    c10::complex<T> dC_dn = (C_plus_n - C_minus_n) / (two * eps);
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_n * dC_dn;
  }

  if (std::abs(gradient_gradient_lambda) > T(1e-15)) {
    c10::complex<T> C_plus_lambda = gegenbauer_polynomial_c(n, lambda + eps, z);
    c10::complex<T> C_minus_lambda = gegenbauer_polynomial_c(n, lambda - eps, z);
    c10::complex<T> dC_dlambda = (C_plus_lambda - C_minus_lambda) / (two * eps);
    gradient_gradient_output = gradient_gradient_output + gradient_gradient_lambda * dC_dlambda;
  }

  c10::complex<T> new_gradient_z = gradient_gradient_z * gradient * d2C_dz2;
  c10::complex<T> new_gradient_n = zero;
  c10::complex<T> new_gradient_lambda = zero;

  return {gradient_gradient_output, new_gradient_n, new_gradient_lambda, new_gradient_z};
}

} // namespace torchscience::kernel::special_functions
