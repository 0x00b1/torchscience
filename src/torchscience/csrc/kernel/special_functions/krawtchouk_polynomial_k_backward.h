#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "krawtchouk_polynomial_k.h"

namespace torchscience::kernel::special_functions {

// Backward for Krawtchouk polynomial K_n(x; p, N)
//
// The Krawtchouk polynomials are discrete orthogonal polynomials.
// All derivatives (w.r.t. n, x, p, N) are computed via finite differences
// since the analytical derivatives involve complex expressions.
template <typename T>
std::tuple<T, T, T, T> krawtchouk_polynomial_k_backward(
    T gradient,
    T n,
    T x,
    T p,
    T N
) {
  T eps = T(1e-7);

  // dK/dn via finite differences
  T K_plus_n = krawtchouk_polynomial_k(n + eps, x, p, N);
  T K_minus_n = krawtchouk_polynomial_k(n - eps, x, p, N);
  T gradient_n = gradient * (K_plus_n - K_minus_n) / (T(2) * eps);

  // dK/dx via finite differences
  T K_plus_x = krawtchouk_polynomial_k(n, x + eps, p, N);
  T K_minus_x = krawtchouk_polynomial_k(n, x - eps, p, N);
  T gradient_x = gradient * (K_plus_x - K_minus_x) / (T(2) * eps);

  // dK/dp via finite differences
  T K_plus_p = krawtchouk_polynomial_k(n, x, p + eps, N);
  T K_minus_p = krawtchouk_polynomial_k(n, x, p - eps, N);
  T gradient_p = gradient * (K_plus_p - K_minus_p) / (T(2) * eps);

  // dK/dN via finite differences
  T K_plus_N = krawtchouk_polynomial_k(n, x, p, N + eps);
  T K_minus_N = krawtchouk_polynomial_k(n, x, p, N - eps);
  T gradient_N = gradient * (K_plus_N - K_minus_N) / (T(2) * eps);

  return {gradient_n, gradient_x, gradient_p, gradient_N};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
krawtchouk_polynomial_k_backward(
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> p,
    c10::complex<T> N
) {
  c10::complex<T> eps(T(1e-7), T(0));
  c10::complex<T> two(T(2), T(0));

  // dK/dn via finite differences
  c10::complex<T> K_plus_n = krawtchouk_polynomial_k(n + eps, x, p, N);
  c10::complex<T> K_minus_n = krawtchouk_polynomial_k(n - eps, x, p, N);
  c10::complex<T> gradient_n = gradient * (K_plus_n - K_minus_n) / (two * eps);

  // dK/dx via finite differences
  c10::complex<T> K_plus_x = krawtchouk_polynomial_k(n, x + eps, p, N);
  c10::complex<T> K_minus_x = krawtchouk_polynomial_k(n, x - eps, p, N);
  c10::complex<T> gradient_x = gradient * (K_plus_x - K_minus_x) / (two * eps);

  // dK/dp via finite differences
  c10::complex<T> K_plus_p = krawtchouk_polynomial_k(n, x, p + eps, N);
  c10::complex<T> K_minus_p = krawtchouk_polynomial_k(n, x, p - eps, N);
  c10::complex<T> gradient_p = gradient * (K_plus_p - K_minus_p) / (two * eps);

  // dK/dN via finite differences
  c10::complex<T> K_plus_N = krawtchouk_polynomial_k(n, x, p, N + eps);
  c10::complex<T> K_minus_N = krawtchouk_polynomial_k(n, x, p, N - eps);
  c10::complex<T> gradient_N = gradient * (K_plus_N - K_minus_N) / (two * eps);

  return {gradient_n, gradient_x, gradient_p, gradient_N};
}

} // namespace torchscience::kernel::special_functions
