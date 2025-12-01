#pragma once

#include <cmath>
#include <limits>
#include "f_cumulative_distribution.h"
#include "f_probability_density.h"

namespace torchscience::kernel::probability {

// F-distribution quantile function (inverse CDF)
// Uses bisection + Newton-Raphson for robustness
template <typename T>
T f_quantile(T p, T d1, T d2, int max_iter = 100, T tol = T(1e-10)) {
  // Edge cases
  if (p <= T(0)) return T(0);
  if (p >= T(1)) return std::numeric_limits<T>::infinity();
  if (std::isnan(p) || std::isnan(d1) || std::isnan(d2)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Initial guess using approximate quantile
  // For F distribution, median is approximately 1 when d1 = d2
  // Use a rough approximation based on p
  T x;
  if (p < T(0.5)) {
    x = T(0.5);
  } else {
    x = T(2.0);
  }

  // Bisection bounds
  T lo = T(0);
  T hi = T(1000);  // Upper bound for search

  // First, expand hi if needed
  while (f_cumulative_distribution(hi, d1, d2) < p && hi < T(1e10)) {
    hi *= T(10);
  }

  // Bisection to get close to the root
  for (int i = 0; i < 30; ++i) {
    T mid = (lo + hi) / T(2);
    T cdf_mid = f_cumulative_distribution(mid, d1, d2);
    if (cdf_mid < p) {
      lo = mid;
    } else {
      hi = mid;
    }
  }

  x = (lo + hi) / T(2);

  // Newton-Raphson refinement
  for (int i = 0; i < max_iter; ++i) {
    T cdf = f_cumulative_distribution(x, d1, d2);
    T pdf = f_probability_density(x, d1, d2);

    // Avoid division by very small values
    if (pdf < T(1e-15)) {
      // Fall back to bisection step
      if (cdf < p) {
        lo = x;
      } else {
        hi = x;
      }
      x = (lo + hi) / T(2);
      continue;
    }

    T delta = (cdf - p) / pdf;
    T x_new = x - delta;

    // Ensure we stay within bounds
    if (x_new <= lo || x_new >= hi) {
      // Fall back to bisection
      if (cdf < p) {
        lo = x;
      } else {
        hi = x;
      }
      x_new = (lo + hi) / T(2);
    } else {
      // Update bounds based on current position
      if (cdf < p) {
        lo = x;
      } else {
        hi = x;
      }
    }

    x = x_new;

    // Check convergence
    if (std::abs(delta) < tol * std::max(T(1), x)) break;
  }

  return x;
}

}  // namespace torchscience::kernel::probability
