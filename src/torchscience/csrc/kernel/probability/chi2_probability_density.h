#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Chi-squared PDF:
// f(x; k) = x^(k/2 - 1) * exp(-x/2) / (2^(k/2) * Gamma(k/2))
// Using log form for numerical stability
template <typename T>
T chi2_probability_density(T x, T df) {
  if (x < T(0)) return T(0);
  if (x == T(0)) {
    // PDF at 0 depends on df: 0 if df >= 2, +inf if df < 2
    if (df < T(2)) return std::numeric_limits<T>::infinity();
    if (df == T(2)) return T(0.5);
    return T(0);
  }
  T k_half = df / T(2);
  T log_probability_density = (k_half - T(1)) * std::log(x) - x / T(2) - k_half * std::log(T(2)) - std::lgamma(k_half);
  return std::exp(log_probability_density);
}

}  // namespace torchscience::kernel::probability
