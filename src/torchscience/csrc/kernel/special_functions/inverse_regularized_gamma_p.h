#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

#include "regularized_gamma_p.h"
#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute the inverse of the standard normal CDF (probit function)
// using the Abramowitz and Stegun approximation
template <typename T>
T inverse_normal_cdf(T p) {
  // Handle edge cases
  if (p <= T(0)) return -std::numeric_limits<T>::infinity();
  if (p >= T(1)) return std::numeric_limits<T>::infinity();

  // Use symmetry: if p > 0.5, compute for 1-p and negate
  bool negate = false;
  if (p > T(0.5)) {
    p = T(1) - p;
    negate = true;
  }

  // Rational approximation for the inverse error function
  // Abramowitz and Stegun 26.2.23
  T t = std::sqrt(-T(2) * std::log(p));

  T c0 = T(2.515517);
  T c1 = T(0.802853);
  T c2 = T(0.010328);
  T d1 = T(1.432788);
  T d2 = T(0.189269);
  T d3 = T(0.001308);

  T result = t - (c0 + c1 * t + c2 * t * t) /
                 (T(1) + d1 * t + d2 * t * t + d3 * t * t * t);

  return negate ? result : -result;
}

template <typename T>
T inverse_regularized_gamma_p_initial_guess(T a, T y) {
  // Initial approximation for solving P(a, x) = y for x
  //
  // Based on:
  // 1. DiDonato and Morris (1986) "Computation of the Incomplete Gamma
  //    Function Ratios and their Inverse"
  // 2. Gil, Segura, Temme (2012) "Efficient and Accurate Algorithms for
  //    the Computation and Inversion of the Incomplete Gamma Function Ratios"

  const T eps = std::numeric_limits<T>::epsilon();

  // Handle extreme cases
  if (y < eps) {
    // Very small y: use leading term of series
    // P(a, x) ~ x^a / (a * Gamma(a)) for small x
    // x ~ (y * Gamma(a+1))^(1/a)
    T log_guess = (std::log(y) + std::lgamma(a + T(1))) / a;
    return std::exp(log_guess);
  }

  if (y > T(1) - eps) {
    // y close to 1: use asymptotic for large x
    // Q(a, x) = 1 - P(a, x) ~ x^(a-1) * e^(-x) / Gamma(a) for large x
    // Rough estimate: x ~ a + sqrt(2*a) * Phi_inv(y)
    T z = inverse_normal_cdf(y);
    return std::max(a + std::sqrt(T(2) * a) * z, a * T(2));
  }

  // For small a, use the series inversion formula
  if (a < T(1)) {
    // x ~ (y * Gamma(a+1))^(1/a)
    T log_guess = (std::log(y) + std::lgamma(a + T(1))) / a;
    T guess = std::exp(log_guess);
    // Ensure positive and not too large
    return std::max(eps, std::min(guess, a * T(50) + T(10)));
  }

  // For moderate to large a, use the normal approximation
  // The gamma distribution approaches normal for large a:
  // X ~ N(a, sqrt(a)) approximately
  // P(a, x) ~ Phi((x - a) / sqrt(a))
  // So x ~ a + sqrt(a) * Phi_inv(y)

  T z = inverse_normal_cdf(y);

  // Wilson-Hilferty transformation is more accurate for a >= 1
  // (x/a)^(1/3) ~ N(1 - 1/(9a), sqrt(1/(9a)))
  // x ~ a * (1 - 1/(9a) + z * sqrt(1/(9a)))^3

  T h = T(1) / (T(9) * a);
  T w = T(1) - h + z * std::sqrt(h);

  // If w would give negative or very small result, use simple normal approx
  if (w <= T(0.1)) {
    // Fall back to normal approximation
    T guess = a + std::sqrt(a) * z;
    return std::max(eps, guess);
  }

  T guess = a * w * w * w;
  return std::max(eps, guess);
}

} // namespace detail

template <typename T>
T inverse_regularized_gamma_p(T a, T y) {
  // Solve P(a, x) = y for x using Halley's method
  // P(a, x) = regularized lower incomplete gamma
  //
  // Reference: "Computation of the Incomplete Gamma Function Ratios
  // and their Inverse" by DiDonato and Morris (1986)

  // Edge cases
  if (std::isnan(a) || std::isnan(y)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (a <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (y <= T(0)) {
    return T(0);
  }

  if (y >= T(1)) {
    return std::numeric_limits<T>::infinity();
  }

  // Initial guess
  T x = detail::inverse_regularized_gamma_p_initial_guess(a, y);

  // Guard against invalid initial guess
  if (x <= T(0) || !std::isfinite(x)) {
    x = a;  // Use a as fallback
  }

  // Precompute log(Gamma(a)) for efficiency
  T log_gamma_a = std::lgamma(a);

  // Halley iteration
  // For f(x) = P(a, x) - y, we want x such that f(x) = 0
  //
  // f'(x) = dP/dx = x^(a-1) * e^(-x) / Gamma(a)
  // f''(x) = ((a-1)/x - 1) * f'(x)
  //
  // Halley's method:
  // x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')

  const int max_iter = 100;
  const T tol = std::numeric_limits<T>::epsilon() * T(4);
  const T min_step = std::numeric_limits<T>::epsilon() * T(10);

  T prev_x = x;
  for (int iter = 0; iter < max_iter; ++iter) {
    // Compute P(a, x)
    T p = regularized_gamma_p(a, x);
    T f = p - y;

    // Check convergence
    if (std::abs(f) < tol) {
      break;
    }

    // Compute f'(x) = x^(a-1) * e^(-x) / Gamma(a)
    // Use log form for numerical stability
    T log_fp = (a - T(1)) * std::log(x) - x - log_gamma_a;
    T fp = std::exp(log_fp);

    // Guard against zero derivative
    if (fp < std::numeric_limits<T>::min() * T(1e10)) {
      // Derivative is too small - try bisection-like step
      if (f > T(0)) {
        // Current P > target, need smaller x
        x = x * T(0.5);
      } else {
        // Current P < target, need larger x
        x = x * T(2);
      }
      continue;
    }

    // Compute f''(x) = ((a-1)/x - 1) * f'(x)
    T fpp = ((a - T(1)) / x - T(1)) * fp;

    // Halley's method: delta = 2*f*f' / (2*f'^2 - f*f'')
    T denom = T(2) * fp * fp - f * fpp;

    T delta;
    if (std::abs(denom) > std::numeric_limits<T>::min() * T(1e10)) {
      delta = T(2) * f * fp / denom;
    } else {
      // Fall back to Newton's method
      delta = f / fp;
    }

    // Limit step size to avoid divergence
    T max_delta = x * T(0.75);  // Don't move more than 75% of current x
    if (delta > max_delta) {
      delta = max_delta;
    } else if (delta < -max_delta) {
      delta = -max_delta;
    }

    // Update
    T new_x = x - delta;

    // Ensure x stays positive
    if (new_x <= T(0)) {
      // Bisect toward zero
      new_x = x * T(0.5);
    }

    // Check for convergence
    if (std::abs(new_x - x) < min_step * x) {
      x = new_x;
      break;
    }

    // Check for oscillation
    if (iter > 10 && std::abs(new_x - prev_x) < min_step * T(10)) {
      // Oscillating - take average
      x = (x + new_x) / T(2);
      break;
    }

    prev_x = x;
    x = new_x;
  }

  return x;
}

} // namespace torchscience::kernel::special_functions
