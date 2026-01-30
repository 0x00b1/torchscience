#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "zeta_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative of zeta function: d^2/ds^2 zeta(s) = sum_{n=2}^inf (ln n)^2 / n^s
template <typename T>
T zeta_second_derivative(T s) {
  if (std::isnan(s)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (s <= T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (s > T(50)) {
    return T(0);
  }

  int N;
  if (s < T(5)) {
    N = 25;
  } else if (s < T(10)) {
    N = 20;
  } else {
    N = 15;
  }

  // d^2/ds^2 zeta(s) = sum_{n=2}^{N-1} (ln n)^2 / n^s + corrections
  T sum = T(0);
  for (int n = 2; n < N; ++n) {
    T ln_n = std::log(static_cast<T>(n));
    sum += ln_n * ln_n * std::pow(static_cast<T>(n), -s);
  }

  // Second derivative of integral term: d^2/ds^2 [N^(1-s)/(s-1)]
  // = N^(1-s) * (ln N)^2 / (s-1) + 2*N^(1-s)*ln(N)/(s-1)^2 + 2*N^(1-s)/(s-1)^3
  T ln_N = std::log(static_cast<T>(N));
  T N_pow = std::pow(static_cast<T>(N), T(1) - s);
  T s_minus_1 = s - T(1);
  T s_minus_1_sq = s_minus_1 * s_minus_1;
  T s_minus_1_cubed = s_minus_1_sq * s_minus_1;

  sum += N_pow * ln_N * ln_N / s_minus_1 +
         T(2) * N_pow * ln_N / s_minus_1_sq +
         T(2) * N_pow / s_minus_1_cubed;

  // Second derivative of boundary term: d^2/ds^2 [1/(2*N^s)] = (ln N)^2/(2*N^s)
  T N_pow_s = std::pow(static_cast<T>(N), -s);
  sum += ln_N * ln_N * N_pow_s / T(2);

  return sum;
}

template <typename T>
c10::complex<T> zeta_second_derivative(c10::complex<T> s) {
  if (std::isnan(s.real()) || std::isnan(s.imag())) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  if (s.real() <= T(1)) {
    return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(),
                           std::numeric_limits<T>::quiet_NaN());
  }

  if (s.real() > T(50)) {
    return c10::complex<T>(T(0), T(0));
  }

  int N = 20;
  if (s.real() >= T(10)) {
    N = 15;
  }

  c10::complex<T> sum(T(0), T(0));
  for (int n = 2; n < N; ++n) {
    T ln_n = std::log(static_cast<T>(n));
    sum += c10::complex<T>(ln_n * ln_n, T(0)) *
           std::pow(c10::complex<T>(static_cast<T>(n), T(0)), -s);
  }

  T ln_N = std::log(static_cast<T>(N));
  c10::complex<T> N_complex(static_cast<T>(N), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> N_pow = std::pow(N_complex, one - s);
  c10::complex<T> s_minus_1 = s - one;
  c10::complex<T> s_minus_1_sq = s_minus_1 * s_minus_1;
  c10::complex<T> s_minus_1_cubed = s_minus_1_sq * s_minus_1;

  sum += N_pow * c10::complex<T>(ln_N * ln_N, T(0)) / s_minus_1 +
         c10::complex<T>(T(2), T(0)) * N_pow * c10::complex<T>(ln_N, T(0)) / s_minus_1_sq +
         c10::complex<T>(T(2), T(0)) * N_pow / s_minus_1_cubed;

  c10::complex<T> N_pow_s = std::pow(N_complex, -s);
  sum += c10::complex<T>(ln_N * ln_N, T(0)) * N_pow_s / c10::complex<T>(T(2), T(0));

  return sum;
}

} // namespace detail

// Second-order backward pass
template <typename T>
std::tuple<T, T> zeta_backward_backward(
    T gradient_gradient,
    T gradient,
    T s) {
  T zeta_prime = detail::zeta_derivative(s);
  T zeta_double_prime = detail::zeta_second_derivative(s);

  // grad_gradient: derivative w.r.t incoming gradient
  T grad_gradient = gradient_gradient * zeta_prime;

  // grad_s: derivative w.r.t s (second derivative)
  T grad_s = gradient_gradient * gradient * zeta_double_prime;

  return {grad_gradient, grad_s};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> zeta_backward_backward(
    c10::complex<T> gradient_gradient,
    c10::complex<T> gradient,
    c10::complex<T> s) {
  c10::complex<T> zeta_prime = detail::zeta_derivative(s);
  c10::complex<T> zeta_double_prime = detail::zeta_second_derivative(s);

  // PyTorch convention with conjugation
  c10::complex<T> grad_gradient = gradient_gradient * std::conj(zeta_prime);
  c10::complex<T> grad_s = gradient_gradient * gradient * std::conj(zeta_double_prime);

  return {grad_gradient, grad_s};
}

} // namespace torchscience::kernel::special_functions
