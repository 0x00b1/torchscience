#pragma once

#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a)
// Uses series expansion for x < a + 1, continued fraction otherwise

template <typename T>
T regularized_gamma_p_series(T a, T x, int max_iter = 200) {
  // Series: P(a, x) = exp(-x) * x^a / Gamma(a) * sum_{n=0}^inf x^n / (a)_{n+1}
  // where (a)_n is the Pochhammer symbol (rising factorial)
  if (x == T(0)) return T(0);

  T term = T(1) / a;
  T sum = term;
  for (int n = 1; n < max_iter; ++n) {
    term *= x / (a + T(n));
    sum += term;
    if (std::abs(term) < std::abs(sum) * std::numeric_limits<T>::epsilon()) {
      break;
    }
  }
  return sum * std::exp(-x + a * std::log(x) - std::lgamma(a));
}

template <typename T>
T regularized_gamma_q_cf(T a, T x, int max_iter = 200) {
  // Continued fraction for Q(a, x) using Lentz's method
  // Q(a, x) = exp(-x) * x^a / Gamma(a) * CF
  T tiny = std::numeric_limits<T>::min() * T(1e10);
  T eps = std::numeric_limits<T>::epsilon();

  T b = x + T(1) - a;
  T c = T(1) / tiny;
  T d = T(1) / b;
  T h = d;

  for (int n = 1; n <= max_iter; ++n) {
    T an = -T(n) * (T(n) - a);
    b += T(2);
    d = an * d + b;
    if (std::abs(d) < tiny) d = tiny;
    c = b + an / c;
    if (std::abs(c) < tiny) c = tiny;
    d = T(1) / d;
    T delta = d * c;
    h *= delta;
    if (std::abs(delta - T(1)) < eps) break;
  }

  return std::exp(-x + a * std::log(x) - std::lgamma(a)) * h;
}

template <typename T>
T regularized_gamma_p(T a, T x) {
  if (x < T(0) || a <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (x == T(0)) return T(0);

  // Choose method based on convergence speed
  if (x < a + T(1)) {
    return regularized_gamma_p_series(a, x);
  } else {
    // P(a, x) = 1 - Q(a, x)
    return T(1) - regularized_gamma_q_cf(a, x);
  }
}

}  // namespace torchscience::kernel::special_functions
