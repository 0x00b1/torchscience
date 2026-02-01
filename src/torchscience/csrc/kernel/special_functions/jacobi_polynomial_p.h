#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "hypergeometric_2_f_1.h"
#include "gamma.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

// Jacobi polynomial P_n^(alpha,beta)(z)
// P_n^(alpha,beta)(z) = Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1)) * 2F1(-n, n+alpha+beta+1; alpha+1; (1-z)/2)
//
// Special cases:
// - P_0^(alpha,beta)(z) = 1
// - P_n^(0,0)(z) = P_n(z) (Legendre polynomial)
// - P_n^(-1/2,-1/2)(z) = (const) * T_n(z) (Chebyshev polynomial of the first kind)
// - P_n^(1/2,1/2)(z) = (const) * U_n(z) (Chebyshev polynomial of the second kind)
// - When alpha = beta, proportional to Gegenbauer polynomials
//
// The recurrence relation is:
// 2n(n+alpha+beta)(2n+alpha+beta-2) P_n^(alpha,beta)(z) =
//   (2n+alpha+beta-1)[(2n+alpha+beta)(2n+alpha+beta-2)z + alpha^2 - beta^2] P_{n-1}^(alpha,beta)(z)
//   - 2(n+alpha-1)(n+beta-1)(2n+alpha+beta) P_{n-2}^(alpha,beta)(z)
//
// Applications:
// - Most general classical orthogonal polynomial on [-1, 1]
// - Weight function (1-z)^alpha * (1+z)^beta
// - Spectral methods, quadrature rules

template <typename T>
T jacobi_polynomial_p(T n, T alpha, T beta, T z) {
  // P_0^(alpha,beta)(z) = 1
  if (std::abs(n) < T(1e-10)) {
    return T(1);
  }

  // For non-negative integer n, use recurrence for better numerical stability
  // when n is small
  T n_val = n;
  if (n_val > T(0) && std::abs(n_val - std::round(static_cast<double>(n_val))) < T(1e-10)) {
    int n_int = static_cast<int>(std::round(static_cast<double>(n_val)));

    if (n_int <= 20) {
      // Use recurrence relation for small integer n
      // P_0^(alpha,beta) = 1
      // P_1^(alpha,beta) = (alpha - beta)/2 + (alpha + beta + 2)*z/2

      if (n_int == 0) {
        return T(1);
      }

      T a = alpha;
      T b = beta;

      if (n_int == 1) {
        return (a - b) / T(2) + (a + b + T(2)) * z / T(2);
      }

      T P_prev = T(1);  // P_0
      T P_curr = (a - b) / T(2) + (a + b + T(2)) * z / T(2);  // P_1

      for (int k = 2; k <= n_int; ++k) {
        T k_val = static_cast<T>(k);
        // Coefficients for the three-term recurrence
        T c0 = T(2) * k_val * (k_val + a + b) * (T(2) * k_val + a + b - T(2));
        T c1 = (T(2) * k_val + a + b - T(1)) * (a * a - b * b);
        T c2 = (T(2) * k_val + a + b - T(2)) * (T(2) * k_val + a + b - T(1)) * (T(2) * k_val + a + b);
        T c3 = T(2) * (k_val + a - T(1)) * (k_val + b - T(1)) * (T(2) * k_val + a + b);

        T P_next = ((c1 + c2 * z) * P_curr - c3 * P_prev) / c0;
        P_prev = P_curr;
        P_curr = P_next;
      }

      return P_curr;
    }
  }

  // For larger n or non-integer n, use the hypergeometric representation
  // P_n^(alpha,beta)(z) = Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1)) * 2F1(-n, n+alpha+beta+1; alpha+1; (1-z)/2)

  // Compute the coefficient: Gamma(n+alpha+1) / (Gamma(alpha+1) * Gamma(n+1))
  T log_coeff = log_gamma(n + alpha + T(1)) - log_gamma(alpha + T(1)) - log_gamma(n + T(1));
  T coeff = std::exp(log_coeff);

  // Handle case where coefficient might be NaN due to gamma poles
  if (!std::isfinite(static_cast<double>(coeff))) {
    coeff = gamma(n + alpha + T(1)) / (gamma(alpha + T(1)) * gamma(n + T(1)));
  }

  // Compute 2F1(-n, n+alpha+beta+1; alpha+1; (1-z)/2)
  T a = -n;
  T b = n + alpha + beta + T(1);
  T c = alpha + T(1);
  T w = (T(1) - z) / T(2);
  T F = hypergeometric_2_f_1(a, b, c, w);

  return coeff * F;
}

// Complex version
template <typename T>
c10::complex<T> jacobi_polynomial_p(c10::complex<T> n, c10::complex<T> alpha, c10::complex<T> beta, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  // P_0^(alpha,beta)(z) = 1
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

        c10::complex<T> a = alpha;
        c10::complex<T> b = beta;

        if (n_int == 1) {
          return (a - b) / two + (a + b + two) * z / two;
        }

        c10::complex<T> P_prev = one;  // P_0
        c10::complex<T> P_curr = (a - b) / two + (a + b + two) * z / two;  // P_1

        for (int k = 2; k <= n_int; ++k) {
          c10::complex<T> k_val(static_cast<T>(k), T(0));
          // Coefficients for the three-term recurrence
          c10::complex<T> c0 = two * k_val * (k_val + a + b) * (two * k_val + a + b - two);
          c10::complex<T> c1 = (two * k_val + a + b - one) * (a * a - b * b);
          c10::complex<T> c2 = (two * k_val + a + b - two) * (two * k_val + a + b - one) * (two * k_val + a + b);
          c10::complex<T> c3 = two * (k_val + a - one) * (k_val + b - one) * (two * k_val + a + b);

          c10::complex<T> P_next = ((c1 + c2 * z) * P_curr - c3 * P_prev) / c0;
          P_prev = P_curr;
          P_curr = P_next;
        }

        return P_curr;
      }
    }
  }

  // Use the hypergeometric representation
  c10::complex<T> log_coeff = log_gamma(n + alpha + one) - log_gamma(alpha + one) - log_gamma(n + one);
  c10::complex<T> coeff = std::exp(log_coeff);

  c10::complex<T> a = -n;
  c10::complex<T> b = n + alpha + beta + one;
  c10::complex<T> c = alpha + one;
  c10::complex<T> w = (one - z) / two;
  c10::complex<T> F = hypergeometric_2_f_1(a, b, c, w);

  return coeff * F;
}

} // namespace torchscience::kernel::special_functions
