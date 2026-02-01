#pragma once

#include <cmath>
#include <limits>

#include "legendre_polynomial_p.h"

namespace torchscience::kernel::special_functions {

// Legendre function of the second kind Q_n(x)
//
// For |x| < 1 and integer n >= 0:
//   Q_0(x) = (1/2) * ln((1+x)/(1-x)) = arctanh(x)
//   Q_1(x) = x * Q_0(x) - 1
//   (n+1) * Q_{n+1}(x) = (2n+1) * x * Q_n(x) - n * Q_{n-1}(x)
//
// Q_n has logarithmic singularities at x = +/- 1.
//
// For non-integer n, we use:
//   Q_n(x) = (pi/2) * (P_n(x) * cos(n*pi) - P_n(-x)) / sin(n*pi)
// where P_n is the Legendre polynomial of the first kind.
// This is undefined at integer n (0/0 form).
template <typename T>
T legendre_polynomial_q(T x, T n) {
  // Base cases using recurrence
  // Q_0(x) = arctanh(x) = (1/2) * ln((1+x)/(1-x))
  T one = T(1);
  T half = T(0.5);
  T pi = T(3.14159265358979323846);

  // Check if n is close to a non-negative integer
  T n_floor = std::floor(n + T(0.5));
  T n_diff = std::abs(n - n_floor);

  if (n_diff < T(1e-10) && n_floor >= T(0)) {
    // Integer n case: use recurrence relation
    int n_int = static_cast<int>(n_floor);

    // Handle singularity at x = +/- 1
    T x_safe = x;
    T eps = T(1e-15);
    if (std::abs(x - one) < eps) {
      return std::numeric_limits<T>::infinity();
    }
    if (std::abs(x + one) < eps) {
      return -std::numeric_limits<T>::infinity();
    }

    // Q_0(x) = (1/2) * ln((1+x)/(1-x)) = arctanh(x)
    T Q0 = half * std::log((one + x_safe) / (one - x_safe));

    if (n_int == 0) {
      return Q0;
    }

    // Q_1(x) = x * Q_0(x) - 1
    T Q1 = x_safe * Q0 - one;

    if (n_int == 1) {
      return Q1;
    }

    // Use recurrence: (n+1)*Q_{n+1} = (2n+1)*x*Q_n - n*Q_{n-1}
    T Q_prev = Q0;
    T Q_curr = Q1;

    for (int k = 1; k < n_int; ++k) {
      T k_t = T(k);
      T Q_next = ((T(2) * k_t + one) * x_safe * Q_curr - k_t * Q_prev) / (k_t + one);
      Q_prev = Q_curr;
      Q_curr = Q_next;
    }

    return Q_curr;
  } else {
    // Non-integer n: use formula with P_n
    // Q_n(x) = (pi/2) * (P_n(x)*cos(n*pi) - P_n(-x)) / sin(n*pi)
    T Pn_x = legendre_polynomial_p(n, x);
    T Pn_neg_x = legendre_polynomial_p(n, -x);
    T cos_n_pi = std::cos(n * pi);
    T sin_n_pi = std::sin(n * pi);

    // Avoid division by zero (should not happen for non-integer n)
    if (std::abs(sin_n_pi) < T(1e-15)) {
      return std::numeric_limits<T>::quiet_NaN();
    }

    return (half * pi) * (Pn_x * cos_n_pi - Pn_neg_x) / sin_n_pi;
  }
}

} // namespace torchscience::kernel::special_functions
