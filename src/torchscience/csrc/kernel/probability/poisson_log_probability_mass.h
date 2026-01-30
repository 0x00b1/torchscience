#pragma once

#include <cmath>

#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::probability {

// Poisson log probability mass function: log(P(X = k)) = k * log(rate) - rate - log_gamma(k + 1)
template <typename T>
T poisson_log_probability_mass(T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0)) return -std::numeric_limits<T>::infinity();

  // Handle rate <= 0 edge case
  if (rate <= T(0)) {
    return (k == T(0)) ? T(0) : -std::numeric_limits<T>::infinity();
  }

  // log(PMF) = k * log(rate) - rate - log_gamma(k+1)
  return k * std::log(rate) - rate - special_functions::log_gamma(k + T(1));
}

}  // namespace torchscience::kernel::probability
