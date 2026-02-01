#pragma once

#include <cmath>
#include <limits>

#include "incomplete_beta.h"
#include "log_beta.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Initial guess for inverse regularized incomplete beta
// Based on algorithms from:
// - TOMS Algorithm 724 by Cran et al.
// - DiDonato and Morris (1992) "Algorithm 708: Significant Digit Computation
//   of the Incomplete Beta Function Ratios"
template <typename T>
T inverse_regularized_incomplete_beta_initial_guess(T a, T b, T y) {
  const T eps = std::numeric_limits<T>::epsilon();

  // Handle extreme cases - use power law for small/large y
  // These are more accurate than normal approximation near boundaries
  T threshold_small = T(0.01);  // Use power law for y < 1%
  T threshold_large = T(0.99);  // Use power law for y > 99%

  if (y < threshold_small) {
    // Small y: use leading term approximation
    // I_x(a, b) ~ x^a / (a * B(a,b)) for small x
    // x ~ (y * a * B(a,b))^(1/a)
    T log_guess = (std::log(y) + std::log(a) + log_beta(a, b)) / a;
    T x0 = std::exp(log_guess);
    return std::max(eps, std::min(T(1) - eps, x0));
  }

  if (y > threshold_large) {
    // Large y: use 1 - I_{1-x}(b, a) relationship
    // 1 - y ~ (1-x)^b / (b * B(a,b)) for x close to 1
    // x ~ 1 - ((1-y) * b * B(a,b))^(1/b)
    T log_guess = (std::log(T(1) - y) + std::log(b) + log_beta(a, b)) / b;
    T x0 = T(1) - std::exp(log_guess);
    return std::max(eps, std::min(T(1) - eps, x0));
  }

  // Use normal approximation for I^{-1}(a, b, y)
  // Based on the Wilson-Hilferty type transformation

  // Compute approximate mean and variance of beta distribution
  T mean = a / (a + b);
  T var = a * b / ((a + b) * (a + b) * (a + b + T(1)));
  T stddev = std::sqrt(var);

  // Normal quantile approximation (Abramowitz & Stegun 26.2.23)
  T p = y;
  if (p > T(0.5)) p = T(1) - p;
  T t = std::sqrt(-T(2) * std::log(p));
  T c0 = T(2.515517);
  T c1 = T(0.802853);
  T c2 = T(0.010328);
  T d1 = T(1.432788);
  T d2 = T(0.189269);
  T d3 = T(0.001308);
  T z = t - (c0 + c1 * t + c2 * t * t) /
               (T(1) + d1 * t + d2 * t * t + d3 * t * t * t);
  if (y > T(0.5)) z = -z;

  // Initial guess using normal approximation
  T x0 = mean + z * stddev;

  // Clamp to valid range
  x0 = std::max(eps, std::min(T(1) - eps, x0));

  // Special cases with exact formulas
  if (std::abs(a - T(1)) < eps * T(100)) {
    // For a = 1: I_x(1, b) = 1 - (1-x)^b
    // Inverse: x = 1 - (1-y)^(1/b)
    T x_exact = T(1) - std::pow(T(1) - y, T(1) / b);
    return std::max(eps, std::min(T(1) - eps, x_exact));
  }

  if (std::abs(b - T(1)) < eps * T(100)) {
    // For b = 1: I_x(a, 1) = x^a
    // Inverse: x = y^(1/a)
    T x_exact = std::pow(y, T(1) / a);
    return std::max(eps, std::min(T(1) - eps, x_exact));
  }

  // For small a (but not exactly 1), use leading term approximation for small y
  if (a < T(1) && y < T(0.5)) {
    T log_guess = (std::log(y) + std::log(a) + log_beta(a, b)) / a;
    T x_small_a = std::exp(log_guess);
    if (std::isfinite(x_small_a) && x_small_a > T(0) && x_small_a < T(1)) {
      x0 = std::min(x0, x_small_a);
    }
  }

  // For small b (but not exactly 1), use leading term approximation for large y
  if (b < T(1) && y > T(0.5)) {
    T log_guess = (std::log(T(1) - y) + std::log(b) + log_beta(a, b)) / b;
    T x_small_b = T(1) - std::exp(log_guess);
    if (std::isfinite(x_small_b) && x_small_b > T(0) && x_small_b < T(1)) {
      x0 = std::max(x0, x_small_b);
    }
  }

  return x0;
}

} // namespace detail

template <typename T>
T inverse_regularized_incomplete_beta(T a, T b, T y) {
  // Solve I_x(a, b) = y for x using Halley's method
  // I_x(a, b) = regularized incomplete beta function
  //
  // Reference: "Computation of the Incomplete Beta Function Ratios
  // and their Inverse" by DiDonato and Morris

  // Edge cases
  if (std::isnan(a) || std::isnan(b) || std::isnan(y)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (a <= T(0) || b <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  if (y <= T(0)) {
    return T(0);
  }

  if (y >= T(1)) {
    return T(1);
  }

  // Initial guess
  T x = detail::inverse_regularized_incomplete_beta_initial_guess(a, b, y);

  // Guard against invalid initial guess
  const T eps = std::numeric_limits<T>::epsilon();
  if (x <= T(0) || x >= T(1) || !std::isfinite(x)) {
    x = a / (a + b);  // Use mean as fallback
  }
  x = std::max(eps * T(10), std::min(T(1) - eps * T(10), x));

  // Precompute log(B(a, b)) for efficiency
  T log_beta_ab = log_beta(a, b);

  // Halley iteration
  // For f(x) = I_x(a, b) - y, we want x such that f(x) = 0
  //
  // f'(x) = dI/dx = x^(a-1) * (1-x)^(b-1) / B(a, b)
  // f''(x) = [(a-1)/x - (b-1)/(1-x)] * f'(x)
  //
  // Halley's method:
  // x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')

  const int max_iter = 100;
  const T tol = eps * T(4);
  const T min_step = eps * T(10);

  T prev_x = x;
  for (int iter = 0; iter < max_iter; ++iter) {
    // Compute I_x(a, b)
    T ix = incomplete_beta(x, a, b);
    T f = ix - y;

    // Check convergence
    if (std::abs(f) < tol) {
      break;
    }

    // Compute f'(x) = x^(a-1) * (1-x)^(b-1) / B(a, b)
    // Use log form for numerical stability
    T log_fp = (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x) - log_beta_ab;
    T fp = std::exp(log_fp);

    // Guard against zero derivative
    if (fp < std::numeric_limits<T>::min() * T(1e10)) {
      // Derivative is too small - try bisection-like step
      if (f > T(0)) {
        x = x * T(0.5);
      } else {
        x = (x + T(1)) * T(0.5);
      }
      x = std::max(eps * T(10), std::min(T(1) - eps * T(10), x));
      continue;
    }

    // Compute f''(x) = [(a-1)/x - (b-1)/(1-x)] * f'(x)
    T fpp = ((a - T(1)) / x - (b - T(1)) / (T(1) - x)) * fp;

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
    T max_delta = std::min(x * T(0.75), (T(1) - x) * T(0.75));
    if (delta > max_delta) {
      delta = max_delta;
    } else if (delta < -max_delta) {
      delta = -max_delta;
    }

    // Update
    T new_x = x - delta;

    // Ensure x stays in (0, 1)
    new_x = std::max(eps * T(10), std::min(T(1) - eps * T(10), new_x));

    // Check for convergence
    if (std::abs(new_x - x) < min_step) {
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
