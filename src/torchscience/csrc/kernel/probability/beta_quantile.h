#pragma once

#include <cmath>
#include <algorithm>

#include "beta_cumulative_distribution.h"
#include "beta_probability_density.h"

namespace torchscience::kernel::probability {

// Beta PPF (quantile function) via Newton-Raphson with Halley's correction
// Find x such that CDF(x; a, b) = p
template <typename T>
T beta_quantile(T p, T a, T b, int max_iter = 100, T tol = T(1e-12)) {
  if (p <= T(0)) return T(0);
  if (p >= T(1)) return T(1);

  // Initial guess based on distribution mode/mean
  T x;
  if (a <= T(1) && b <= T(1)) {
    // U-shaped distribution, start at p
    x = p;
  } else if (a > T(1) && b > T(1)) {
    // Bell-shaped, start near mode = (a-1)/(a+b-2)
    T mode = (a - T(1)) / (a + b - T(2));
    x = mode + (p - T(0.5)) * T(0.3);  // Adjust based on p
    x = std::max(T(0.01), std::min(T(0.99), x));
  } else {
    // Skewed, use mean as starting point
    T mean = a / (a + b);
    x = mean;
  }

  // Newton-Raphson iteration with damping
  for (int i = 0; i < max_iter; ++i) {
    T cdf = beta_cumulative_distribution(x, a, b);
    T pdf = beta_probability_density(x, a, b);

    // If pdf is too small, try bisection step instead
    if (pdf < T(1e-15)) {
      // Bisection fallback
      if (cdf < p) {
        x = (x + T(1)) / T(2);
      } else {
        x = x / T(2);
      }
      continue;
    }

    T delta = (cdf - p) / pdf;

    // Damping for stability
    T damping = T(1);
    if (std::abs(delta) > T(0.5)) {
      damping = T(0.5) / std::abs(delta);
    }

    T x_new = x - damping * delta;

    // Clamp to (0, 1)
    x_new = std::max(T(1e-12), std::min(T(1) - T(1e-12), x_new));

    // Check convergence
    if (std::abs(x_new - x) < tol) {
      return x_new;
    }

    x = x_new;
  }

  return x;
}

}  // namespace torchscience::kernel::probability
