#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "meixner_polynomial_m.h"

namespace torchscience::kernel::special_functions {

// Backward for Meixner polynomial M_n(x; beta, c)
//
// The Meixner polynomials are discrete orthogonal polynomials.
// All derivatives (w.r.t. n, x, beta, c) are computed via finite differences
// since the analytical derivatives involve complex expressions.
template <typename T>
std::tuple<T, T, T, T> meixner_polynomial_m_backward(
    T gradient,
    T n,
    T x,
    T beta,
    T c
) {
  T eps = T(1e-7);

  // dM/dn via finite differences
  T M_plus_n = meixner_polynomial_m(n + eps, x, beta, c);
  T M_minus_n = meixner_polynomial_m(n - eps, x, beta, c);
  T gradient_n = gradient * (M_plus_n - M_minus_n) / (T(2) * eps);

  // dM/dx via finite differences
  T M_plus_x = meixner_polynomial_m(n, x + eps, beta, c);
  T M_minus_x = meixner_polynomial_m(n, x - eps, beta, c);
  T gradient_x = gradient * (M_plus_x - M_minus_x) / (T(2) * eps);

  // dM/dbeta via finite differences
  T M_plus_beta = meixner_polynomial_m(n, x, beta + eps, c);
  T M_minus_beta = meixner_polynomial_m(n, x, beta - eps, c);
  T gradient_beta = gradient * (M_plus_beta - M_minus_beta) / (T(2) * eps);

  // dM/dc via finite differences
  T M_plus_c = meixner_polynomial_m(n, x, beta, c + eps);
  T M_minus_c = meixner_polynomial_m(n, x, beta, c - eps);
  T gradient_c = gradient * (M_plus_c - M_minus_c) / (T(2) * eps);

  return {gradient_n, gradient_x, gradient_beta, gradient_c};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
meixner_polynomial_m_backward(
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> x,
    c10::complex<T> beta,
    c10::complex<T> c
) {
  c10::complex<T> eps(T(1e-7), T(0));
  c10::complex<T> two(T(2), T(0));

  // dM/dn via finite differences
  c10::complex<T> M_plus_n = meixner_polynomial_m(n + eps, x, beta, c);
  c10::complex<T> M_minus_n = meixner_polynomial_m(n - eps, x, beta, c);
  c10::complex<T> gradient_n = gradient * (M_plus_n - M_minus_n) / (two * eps);

  // dM/dx via finite differences
  c10::complex<T> M_plus_x = meixner_polynomial_m(n, x + eps, beta, c);
  c10::complex<T> M_minus_x = meixner_polynomial_m(n, x - eps, beta, c);
  c10::complex<T> gradient_x = gradient * (M_plus_x - M_minus_x) / (two * eps);

  // dM/dbeta via finite differences
  c10::complex<T> M_plus_beta = meixner_polynomial_m(n, x, beta + eps, c);
  c10::complex<T> M_minus_beta = meixner_polynomial_m(n, x, beta - eps, c);
  c10::complex<T> gradient_beta = gradient * (M_plus_beta - M_minus_beta) / (two * eps);

  // dM/dc via finite differences
  c10::complex<T> M_plus_c = meixner_polynomial_m(n, x, beta, c + eps);
  c10::complex<T> M_minus_c = meixner_polynomial_m(n, x, beta, c - eps);
  c10::complex<T> gradient_c = gradient * (M_plus_c - M_minus_c) / (two * eps);

  return {gradient_n, gradient_x, gradient_beta, gradient_c};
}

} // namespace torchscience::kernel::special_functions
