#pragma once

#include <cmath>
#include "../special_functions/regularized_gamma_q.h"

namespace torchscience::kernel::probability {

// Poisson CDF: P(X <= k) = Q(k+1, lambda) = 1 - P(k+1, lambda)
// where Q is the upper regularized incomplete gamma function
template <typename T>
T poisson_cumulative_distribution(T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0)) return T(0);

  // CDF = Q(k+1, rate) = regularized_gamma_q(k+1, rate)
  return special_functions::regularized_gamma_q(k + T(1), rate);
}

}  // namespace torchscience::kernel::probability
