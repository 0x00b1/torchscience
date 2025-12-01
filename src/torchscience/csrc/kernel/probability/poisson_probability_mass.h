#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Poisson probability mass function: P(X = k) = lambda^k * exp(-lambda) / k!
// Computed in log space for numerical stability
template <typename T>
T poisson_probability_mass(T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0)) return T(0);

  // Handle rate = 0 edge case
  if (rate <= T(0)) {
    return (k == T(0)) ? T(1) : T(0);
  }

  // PMF = rate^k * exp(-rate) / k!
  // log(PMF) = k * log(rate) - rate - lgamma(k+1)
  T log_pmf = k * std::log(rate) - rate - std::lgamma(k + T(1));

  return std::exp(log_pmf);
}

}  // namespace torchscience::kernel::probability
