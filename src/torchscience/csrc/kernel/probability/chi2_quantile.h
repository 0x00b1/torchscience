#pragma once

#include <cmath>
#include <limits>
#include "chi2_cumulative_distribution.h"
#include "chi2_probability_density.h"
#include "normal_quantile.h"

namespace torchscience::kernel::probability {

// Chi-squared quantile function (inverse CDF)
// Uses Wilson-Hilferty approximation for initial guess + Newton-Raphson
template <typename T>
T chi2_quantile(T p, T df, int max_iter = 50, T tol = T(1e-10)) {
  // Edge cases
  if (p <= T(0)) return T(0);
  if (p >= T(1)) return std::numeric_limits<T>::infinity();
  if (std::isnan(p) || std::isnan(df)) return std::numeric_limits<T>::quiet_NaN();

  // Wilson-Hilferty approximation for initial guess
  T z = standard_normal_quantile(p);
  T h = T(2) / (T(9) * df);
  T x = df * std::pow(T(1) - h + z * std::sqrt(h), T(3));

  // Clamp to positive
  if (x < T(0.01)) x = T(0.01);

  // Newton-Raphson refinement
  for (int i = 0; i < max_iter; ++i) {
    T cdf = chi2_cumulative_distribution(x, df);
    T pdf = chi2_probability_density(x, df);

    // Avoid division by very small values
    if (pdf < T(1e-15)) break;

    T delta = (cdf - p) / pdf;
    x -= delta;

    // Clamp to positive
    if (x < T(0.001)) x = T(0.001);

    // Check convergence
    if (std::abs(delta) < tol * x) break;
  }

  return x;
}

}  // namespace torchscience::kernel::probability
