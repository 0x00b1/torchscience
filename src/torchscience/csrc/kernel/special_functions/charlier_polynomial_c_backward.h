#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "charlier_polynomial_c.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Charlier polynomial C_n(x; a)
//
// We compute gradients with respect to n, x, and a.
//
// For the gradient with respect to x:
// Using the recurrence a * C_{n+1} = (x - n - a) * C_n - n * C_{n-1},
// we can differentiate to get:
// dC_n/dx can be computed via finite differences or using the relation:
//   dC_n/dx = C_{n-1}/a for n >= 1
//
// However, for robustness, we use finite differences for all gradients.

template <typename T>
std::tuple<T, T, T> charlier_polynomial_c_backward(T gradient, T n, T x, T a) {
  T eps = T(1e-7);

  // Gradient with respect to x using finite differences
  T C_x_plus = charlier_polynomial_c(n, x + eps, a);
  T C_x_minus = charlier_polynomial_c(n, x - eps, a);
  T grad_x = (C_x_plus - C_x_minus) / (T(2) * eps);

  // Gradient with respect to n using finite differences
  T C_n_plus = charlier_polynomial_c(n + eps, x, a);
  T C_n_minus = charlier_polynomial_c(n - eps, x, a);
  T grad_n = (C_n_plus - C_n_minus) / (T(2) * eps);

  // Gradient with respect to a using finite differences
  T C_a_plus = charlier_polynomial_c(n, x, a + eps);
  T C_a_minus = charlier_polynomial_c(n, x, a - eps);
  T grad_a = (C_a_plus - C_a_minus) / (T(2) * eps);

  return {gradient * grad_n, gradient * grad_x, gradient * grad_a};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
charlier_polynomial_c_backward(c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> x, c10::complex<T> a) {
  T eps_val = T(1e-7);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> two(T(2), T(0));

  // Gradient with respect to x using finite differences
  c10::complex<T> C_x_plus = charlier_polynomial_c(n, x + eps, a);
  c10::complex<T> C_x_minus = charlier_polynomial_c(n, x - eps, a);
  c10::complex<T> grad_x = (C_x_plus - C_x_minus) / (two * eps);

  // Gradient with respect to n using finite differences
  c10::complex<T> C_n_plus = charlier_polynomial_c(n + eps, x, a);
  c10::complex<T> C_n_minus = charlier_polynomial_c(n - eps, x, a);
  c10::complex<T> grad_n = (C_n_plus - C_n_minus) / (two * eps);

  // Gradient with respect to a using finite differences
  c10::complex<T> C_a_plus = charlier_polynomial_c(n, x, a + eps);
  c10::complex<T> C_a_minus = charlier_polynomial_c(n, x, a - eps);
  c10::complex<T> grad_a = (C_a_plus - C_a_minus) / (two * eps);

  // For complex holomorphic functions, use Wirtinger derivative convention
  return {
    gradient * std::conj(grad_n),
    gradient * std::conj(grad_x),
    gradient * std::conj(grad_a)
  };
}

} // namespace torchscience::kernel::special_functions
