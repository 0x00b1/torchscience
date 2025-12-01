#pragma once

#include <cmath>
#include <algorithm>
#include <limits>

#include "gamma_cumulative_distribution.h"
#include "gamma_probability_density.h"

namespace torchscience::kernel::probability {

// Gamma PPF (quantile function) via Newton-Raphson with bisection fallback
// Find x such that CDF(x; shape, scale) = p
template <typename T>
T gamma_quantile(T p, T shape, T scale, int max_iter = 100, T tol = T(1e-12)) {
  if (p <= T(0)) return T(0);
  if (p >= T(1)) return std::numeric_limits<T>::infinity();

  // Initial guess using Wilson-Hilferty approximation for gamma quantiles
  // For large shape, gamma is approximately normal with mean=shape*scale, var=shape*scale^2
  T mean = shape * scale;
  T std_dev = std::sqrt(shape) * scale;

  // Use chi-squared approximation for initial guess (gamma with scale=2)
  // Wilson-Hilferty: (x/shape)^(1/3) ~ N(1 - 1/(9*shape), sqrt(1/(9*shape)))
  T x;
  if (shape >= T(1)) {
    // Use Wilson-Hilferty transformation
    T z = T(2.5758);  // ~99th percentile of standard normal
    if (p < T(0.5)) z = -z * std::pow(T(1) - p, T(0.2));
    else z = z * std::pow(p, T(0.2));

    T h = T(1) / (T(9) * shape);
    T wh = T(1) - h + z * std::sqrt(h);
    x = shape * wh * wh * wh * scale;
    x = std::max(x, scale * T(0.01));
  } else {
    // For shape < 1, start with a simpler guess
    x = mean * p * T(2);
    x = std::max(x, scale * T(0.01));
  }

  // Newton-Raphson iteration with damping
  T x_lo = T(0);
  T x_hi = mean + std_dev * T(20);  // Upper bound

  for (int i = 0; i < max_iter; ++i) {
    T cdf = gamma_cumulative_distribution(x, shape, scale);
    T pdf = gamma_probability_density(x, shape, scale);

    // Update bounds for bisection fallback
    if (cdf < p) {
      x_lo = x;
    } else {
      x_hi = x;
    }

    // If pdf is too small, use bisection
    if (pdf < T(1e-15)) {
      x = (x_lo + x_hi) / T(2);
      continue;
    }

    T delta = (cdf - p) / pdf;

    // Damping for stability
    T damping = T(1);
    if (std::abs(delta) > x * T(0.5)) {
      damping = x * T(0.5) / std::abs(delta);
    }

    T x_new = x - damping * delta;

    // Keep x positive and within bounds
    x_new = std::max(T(1e-12), x_new);
    x_new = std::min(x_hi, std::max(x_lo, x_new));

    // Check convergence
    if (std::abs(x_new - x) < tol * std::max(T(1), x)) {
      return x_new;
    }

    x = x_new;
  }

  return x;
}

}  // namespace torchscience::kernel::probability
