#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "hahn_polynomial_q.h"

namespace torchscience::kernel::special_functions {

// Backward for Hahn polynomial Q_n(x; alpha, beta, N)
//
// The Hahn polynomials are discrete orthogonal polynomials.
// All derivatives (w.r.t. n, x, alpha, beta, N) are computed via finite differences
// since the analytical derivatives involve complex expressions.
template <typename T>
std::tuple<T, T, T, T, T> hahn_polynomial_q_backward(
    T gradient,
    T n,
    T x,
    T alpha,
    T beta,
    T N
) {
  T eps = T(1e-7);

  // dQ/dn via finite differences
  T Q_plus_n = hahn_polynomial_q(n + eps, x, alpha, beta, N);
  T Q_minus_n = hahn_polynomial_q(n - eps, x, alpha, beta, N);
  T gradient_n = gradient * (Q_plus_n - Q_minus_n) / (T(2) * eps);

  // dQ/dx via finite differences
  T Q_plus_x = hahn_polynomial_q(n, x + eps, alpha, beta, N);
  T Q_minus_x = hahn_polynomial_q(n, x - eps, alpha, beta, N);
  T gradient_x = gradient * (Q_plus_x - Q_minus_x) / (T(2) * eps);

  // dQ/dalpha via finite differences
  T Q_plus_alpha = hahn_polynomial_q(n, x, alpha + eps, beta, N);
  T Q_minus_alpha = hahn_polynomial_q(n, x, alpha - eps, beta, N);
  T gradient_alpha = gradient * (Q_plus_alpha - Q_minus_alpha) / (T(2) * eps);

  // dQ/dbeta via finite differences
  T Q_plus_beta = hahn_polynomial_q(n, x, alpha, beta + eps, N);
  T Q_minus_beta = hahn_polynomial_q(n, x, alpha, beta - eps, N);
  T gradient_beta = gradient * (Q_plus_beta - Q_minus_beta) / (T(2) * eps);

  // dQ/dN via finite differences
  T Q_plus_N = hahn_polynomial_q(n, x, alpha, beta, N + eps);
  T Q_minus_N = hahn_polynomial_q(n, x, alpha, beta, N - eps);
  T gradient_N = gradient * (Q_plus_N - Q_minus_N) / (T(2) * eps);

  return {gradient_n, gradient_x, gradient_alpha, gradient_beta, gradient_N};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
hahn_polynomial_q_backward(
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> alpha,
    c10::complex<T> beta,
    c10::complex<T> N
) {
  c10::complex<T> eps(T(1e-7), T(0));
  c10::complex<T> two(T(2), T(0));

  // dQ/dn via finite differences
  c10::complex<T> Q_plus_n = hahn_polynomial_q(n + eps, x, alpha, beta, N);
  c10::complex<T> Q_minus_n = hahn_polynomial_q(n - eps, x, alpha, beta, N);
  c10::complex<T> gradient_n = gradient * (Q_plus_n - Q_minus_n) / (two * eps);

  // dQ/dx via finite differences
  c10::complex<T> Q_plus_x = hahn_polynomial_q(n, x + eps, alpha, beta, N);
  c10::complex<T> Q_minus_x = hahn_polynomial_q(n, x - eps, alpha, beta, N);
  c10::complex<T> gradient_x = gradient * (Q_plus_x - Q_minus_x) / (two * eps);

  // dQ/dalpha via finite differences
  c10::complex<T> Q_plus_alpha = hahn_polynomial_q(n, x, alpha + eps, beta, N);
  c10::complex<T> Q_minus_alpha = hahn_polynomial_q(n, x, alpha - eps, beta, N);
  c10::complex<T> gradient_alpha = gradient * (Q_plus_alpha - Q_minus_alpha) / (two * eps);

  // dQ/dbeta via finite differences
  c10::complex<T> Q_plus_beta = hahn_polynomial_q(n, x, alpha, beta + eps, N);
  c10::complex<T> Q_minus_beta = hahn_polynomial_q(n, x, alpha, beta - eps, N);
  c10::complex<T> gradient_beta = gradient * (Q_plus_beta - Q_minus_beta) / (two * eps);

  // dQ/dN via finite differences
  c10::complex<T> Q_plus_N = hahn_polynomial_q(n, x, alpha, beta, N + eps);
  c10::complex<T> Q_minus_N = hahn_polynomial_q(n, x, alpha, beta, N - eps);
  c10::complex<T> gradient_N = gradient * (Q_plus_N - Q_minus_N) / (two * eps);

  return {gradient_n, gradient_x, gradient_alpha, gradient_beta, gradient_N};
}

} // namespace torchscience::kernel::special_functions
