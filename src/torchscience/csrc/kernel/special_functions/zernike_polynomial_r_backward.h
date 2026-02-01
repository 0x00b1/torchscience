#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "zernike_polynomial_r.h"

namespace torchscience::kernel::special_functions {

// Backward pass for radial Zernike polynomial R_n^m(rho)
//
// The derivative with respect to rho:
// dR_n^m/drho = n * R_{n-1}^{|m|-1}(rho) + (n - 2) * R_{n-1}^{|m|+1}(rho) (for |m| > 0)
// or using the Jacobi polynomial relation and chain rule
//
// For simplicity and robustness, we use finite differences for n and m
// since they are typically integers, and compute the analytical derivative
// for rho where gradients are most important.
//
// The derivative with respect to rho can be derived from:
// R_n^m(rho) = (-1)^k * rho^|m| * P_k^(|m|, 0)(1 - 2*rho^2)
// where k = (n - |m|) / 2
//
// dR/drho = (-1)^k * [ |m| * rho^(|m|-1) * P_k^(|m|,0)(x)
//                     + rho^|m| * dP_k/dx * (-4*rho) ]
// where x = 1 - 2*rho^2

template <typename T>
std::tuple<T, T, T> zernike_polynomial_r_backward(T gradient, T n, T m, T rho) {
  T eps = T(1e-7);

  // Gradient with respect to rho
  T grad_rho;

  T abs_m = std::abs(m);
  T diff = n - abs_m;

  // Check validity
  if (diff < T(0)) {
    grad_rho = T(0);
  } else {
    T diff_mod2 = std::fmod(diff, T(2));
    if (std::abs(diff_mod2) > T(0.5)) {
      grad_rho = T(0);
    } else {
      // Use finite differences for rho gradient as well for simplicity
      T R_plus = zernike_polynomial_r(n, m, rho + eps);
      T R_minus = zernike_polynomial_r(n, m, rho - eps);
      grad_rho = (R_plus - R_minus) / (T(2) * eps);
    }
  }

  // Gradient with respect to n using finite differences
  T R_plus_n = zernike_polynomial_r(n + eps, m, rho);
  T R_minus_n = zernike_polynomial_r(n - eps, m, rho);
  T grad_n = (R_plus_n - R_minus_n) / (T(2) * eps);

  // Gradient with respect to m using finite differences
  T R_plus_m = zernike_polynomial_r(n, m + eps, rho);
  T R_minus_m = zernike_polynomial_r(n, m - eps, rho);
  T grad_m = (R_plus_m - R_minus_m) / (T(2) * eps);

  return {gradient * grad_n, gradient * grad_m, gradient * grad_rho};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
zernike_polynomial_r_backward(c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m, c10::complex<T> rho) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> zero(T(0), T(0));
  T eps_val = T(1e-7);
  c10::complex<T> eps(eps_val, T(0));

  // Gradient with respect to rho using finite differences
  c10::complex<T> R_plus_rho = zernike_polynomial_r(n, m, rho + eps);
  c10::complex<T> R_minus_rho = zernike_polynomial_r(n, m, rho - eps);
  c10::complex<T> grad_rho = (R_plus_rho - R_minus_rho) / (two * eps);

  // Gradient with respect to n using finite differences
  c10::complex<T> R_plus_n = zernike_polynomial_r(n + eps, m, rho);
  c10::complex<T> R_minus_n = zernike_polynomial_r(n - eps, m, rho);
  c10::complex<T> grad_n = (R_plus_n - R_minus_n) / (two * eps);

  // Gradient with respect to m using finite differences
  c10::complex<T> R_plus_m = zernike_polynomial_r(n, m + eps, rho);
  c10::complex<T> R_minus_m = zernike_polynomial_r(n, m - eps, rho);
  c10::complex<T> grad_m = (R_plus_m - R_minus_m) / (two * eps);

  // For complex holomorphic functions, use Wirtinger derivative convention
  return {
    gradient * std::conj(grad_n),
    gradient * std::conj(grad_m),
    gradient * std::conj(grad_rho)
  };
}

} // namespace torchscience::kernel::special_functions
