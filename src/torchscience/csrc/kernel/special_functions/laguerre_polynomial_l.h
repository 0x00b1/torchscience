#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "confluent_hypergeometric_m.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

// Generalized Laguerre polynomial L_n^alpha(z)
// L_n^alpha(z) = Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1)) * 1F1(-n; alpha+1; z)
//
// Special cases:
// - L_0^alpha(z) = 1
// - L_1^alpha(z) = 1 + alpha - z
// - L_n^0(z) = L_n(z) (ordinary Laguerre polynomial)
//
// The recurrence relation is:
// (n+1) L_{n+1}^alpha(z) = (2n + alpha + 1 - z) L_n^alpha(z) - (n + alpha) L_{n-1}^alpha(z)
//
// Applications:
// - Quantum mechanics: radial wave functions of the hydrogen atom
// - Orthogonal polynomials with weight function x^alpha * exp(-x) on [0, infinity)

template <typename T>
T laguerre_polynomial_l(T n, T alpha, T z) {
  // L_0^alpha(z) = 1
  if (std::abs(n) < T(1e-10)) {
    return T(1);
  }

  // For negative n, use the reflection formula
  // L_{-n}^alpha(z) = L_{n-1}^{alpha}(-z) * exp(-z) * (-1)^{n-1} for integer n
  // For now, just compute directly via the hypergeometric representation

  // Check for invalid parameters
  // Gamma(alpha+1) has a pole when alpha is a negative integer
  // However, these poles may cancel with the ratio

  // Compute the coefficient: Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1))
  // This is the generalized binomial coefficient C(n+alpha, n)

  T coeff;

  // For non-negative integer n, use recurrence for better numerical stability
  // when n is small
  T n_val = n;
  if (n_val > T(0) && std::abs(n_val - std::round(static_cast<double>(n_val))) < T(1e-10)) {
    int n_int = static_cast<int>(std::round(static_cast<double>(n_val)));

    if (n_int <= 20) {
      // Use recurrence relation for small integer n
      // L_0^alpha = 1
      // L_1^alpha = 1 + alpha - z
      // (n+1) L_{n+1}^alpha = (2n + alpha + 1 - z) L_n^alpha - (n + alpha) L_{n-1}^alpha

      if (n_int == 0) {
        return T(1);
      }
      if (n_int == 1) {
        return T(1) + alpha - z;
      }

      T L_prev = T(1);              // L_0
      T L_curr = T(1) + alpha - z;  // L_1

      for (int k = 1; k < n_int; ++k) {
        T k_val = static_cast<T>(k);
        T L_next = ((T(2) * k_val + alpha + T(1) - z) * L_curr - (k_val + alpha) * L_prev) / (k_val + T(1));
        L_prev = L_curr;
        L_curr = L_next;
      }

      return L_curr;
    }
  }

  // For larger n or non-integer n, use the hypergeometric representation
  // L_n^alpha(z) = C(n+alpha, n) * 1F1(-n; alpha+1; z)

  // Compute C(n+alpha, n) = Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1))
  T log_coeff = log_gamma(n + alpha + T(1)) - log_gamma(alpha + T(1)) - log_gamma(n + T(1));
  coeff = std::exp(log_coeff);

  // Handle case where coefficient might be NaN due to gamma poles
  if (!std::isfinite(static_cast<double>(coeff))) {
    // Try computing directly
    coeff = gamma(n + alpha + T(1)) / (gamma(alpha + T(1)) * gamma(n + T(1)));
  }

  // Compute 1F1(-n; alpha+1; z)
  T a = -n;
  T b = alpha + T(1);
  T M = confluent_hypergeometric_m(a, b, z);

  return coeff * M;
}

// Complex version
template <typename T>
c10::complex<T> laguerre_polynomial_l(c10::complex<T> n, c10::complex<T> alpha, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> zero(T(0), T(0));

  // L_0^alpha(z) = 1
  if (std::abs(n) < T(1e-10)) {
    return one;
  }

  // For non-negative integer n, use recurrence for better numerical stability
  if (std::abs(n.imag()) < T(1e-10) && n.real() > T(0)) {
    T n_real = n.real();
    if (std::abs(n_real - std::round(static_cast<double>(n_real))) < T(1e-10)) {
      int n_int = static_cast<int>(std::round(static_cast<double>(n_real)));

      if (n_int <= 20) {
        if (n_int == 0) {
          return one;
        }
        if (n_int == 1) {
          return one + alpha - z;
        }

        c10::complex<T> L_prev = one;              // L_0
        c10::complex<T> L_curr = one + alpha - z;  // L_1

        for (int k = 1; k < n_int; ++k) {
          c10::complex<T> k_val(static_cast<T>(k), T(0));
          c10::complex<T> two(T(2), T(0));
          c10::complex<T> L_next = ((two * k_val + alpha + one - z) * L_curr - (k_val + alpha) * L_prev) / (k_val + one);
          L_prev = L_curr;
          L_curr = L_next;
        }

        return L_curr;
      }
    }
  }

  // Use the hypergeometric representation
  c10::complex<T> log_coeff = log_gamma(n + alpha + one) - log_gamma(alpha + one) - log_gamma(n + one);
  c10::complex<T> coeff = std::exp(log_coeff);

  c10::complex<T> a = -n;
  c10::complex<T> b = alpha + one;
  c10::complex<T> M = confluent_hypergeometric_m(a, b, z);

  return coeff * M;
}

} // namespace torchscience::kernel::special_functions
