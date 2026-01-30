#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

namespace detail {

// Derivative of zeta function: d/ds zeta(s) = -sum_{n=2}^inf ln(n)/n^s
// Also uses Euler-Maclaurin summation
template <typename T>
T zeta_derivative(T s) {
  // Handle special cases
  if (std::isnan(s)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For s <= 1, return NaN
  if (s <= T(1)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // For very large s, derivative -> 0
  if (s > T(50)) {
    return T(0);
  }

  // Direct summation with Euler-Maclaurin correction
  int N;
  if (s < T(5)) {
    N = 25;
  } else if (s < T(10)) {
    N = 20;
  } else {
    N = 15;
  }

  // d/ds zeta(s) = -sum_{n=2}^{N-1} ln(n)/n^s + correction terms
  T sum = T(0);
  for (int n = 2; n < N; ++n) {
    T ln_n = std::log(static_cast<T>(n));
    sum -= ln_n * std::pow(static_cast<T>(n), -s);
  }

  // Integral term derivative: d/ds [N^(1-s)/(s-1)]
  // = -N^(1-s) * ln(N) / (s-1) - N^(1-s) / (s-1)^2
  T ln_N = std::log(static_cast<T>(N));
  T N_pow = std::pow(static_cast<T>(N), T(1) - s);
  T s_minus_1 = s - T(1);
  sum += -N_pow * ln_N / s_minus_1 - N_pow / (s_minus_1 * s_minus_1);

  // Boundary term derivative: d/ds [1/(2*N^s)] = -ln(N)/(2*N^s)
  T N_pow_s = std::pow(static_cast<T>(N), -s);
  sum += -ln_N * N_pow_s / T(2);

  // Derivatives of Euler-Maclaurin correction terms are more complex
  // For simplicity, we use the same Bernoulli-based correction
  // with the derivative formula
  T power = N_pow_s / static_cast<T>(N);

  for (int k = 0; k < 8; ++k) {  // Use fewer terms for derivative
    int two_k = 2 * (k + 1);

    T coeff = bernoulli_2n[k];

    // The derivative involves both the falling factorial derivative
    // and the power term derivative. This is complex, so we use
    // finite differences implicitly via the overall accuracy.
    // We add a simplified correction based on derivative of N^(-s-2k+1):
    T falling = T(1);
    for (int j = 0; j < two_k - 1; ++j) {
      falling *= (s + static_cast<T>(j)) / static_cast<T>(two_k - j);
    }

    // Derivative of falling factorial (digamma-like terms)
    T falling_deriv = T(0);
    for (int j = 0; j < two_k - 1; ++j) {
      T term = T(1);
      for (int i = 0; i < two_k - 1; ++i) {
        if (i != j) {
          term *= (s + static_cast<T>(i)) / static_cast<T>(two_k - i);
        } else {
          term *= T(1) / static_cast<T>(two_k - i);
        }
      }
      falling_deriv += term;
    }

    // Power derivative: d/ds N^(-s-2k+1) = -ln(N) * N^(-s-2k+1)
    T power_deriv = -ln_N * power;

    T correction = coeff * (falling_deriv * power + falling * power_deriv);
    sum += correction;

    power /= static_cast<T>(N * N);

    if (std::abs(correction) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
      break;
    }
  }

  return sum;
}

template <typename T>
c10::complex<T> zeta_derivative(c10::complex<T> s) {
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
    sum -= c10::complex<T>(ln_n, T(0)) *
           std::pow(c10::complex<T>(static_cast<T>(n), T(0)), -s);
  }

  T ln_N = std::log(static_cast<T>(N));
  c10::complex<T> N_complex(static_cast<T>(N), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> N_pow = std::pow(N_complex, one - s);
  c10::complex<T> s_minus_1 = s - one;

  sum += -N_pow * c10::complex<T>(ln_N, T(0)) / s_minus_1 -
         N_pow / (s_minus_1 * s_minus_1);

  c10::complex<T> N_pow_s = std::pow(N_complex, -s);
  sum += -c10::complex<T>(ln_N, T(0)) * N_pow_s / c10::complex<T>(T(2), T(0));

  return sum;
}

} // namespace detail

// Backward pass for zeta: d/ds zeta(s)
template <typename T>
T zeta_backward(T gradient, T s) {
  return gradient * detail::zeta_derivative(s);
}

template <typename T>
c10::complex<T> zeta_backward(c10::complex<T> gradient, c10::complex<T> s) {
  c10::complex<T> deriv = detail::zeta_derivative(s);
  // PyTorch convention for holomorphic functions
  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
