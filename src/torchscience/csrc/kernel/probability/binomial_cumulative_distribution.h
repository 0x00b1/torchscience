#pragma once

#include <cmath>
#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// Binomial CDF: P(X <= k) = I_{1-p}(n-k, k+1) for k in {0, 1, ..., n}
// where I_x(a, b) is the regularized incomplete beta function
template <typename T>
T binomial_cumulative_distribution(T k, T n, T p) {
  // Floor k for non-integer inputs
  k = std::floor(k);

  // Boundary cases
  if (k < T(0)) return T(0);
  if (k >= n) return T(1);

  // CDF = I_{1-p}(n-k, k+1)
  return special_functions::incomplete_beta(T(1) - p, n - k, k + T(1));
}

}  // namespace torchscience::kernel::probability
