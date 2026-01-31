#pragma once

#include <cmath>

#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::probability {

// Binomial log probability mass function:
// log(P(X = k)) = log_gamma(n + 1) - log_gamma(k + 1) - log_gamma(n - k + 1)
//               + k * log(p) + (n - k) * log(1 - p)
template <typename T>
T binomial_log_probability_mass(T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases: k < 0 or k > n returns -inf
  if (k < T(0) || k > n) return -std::numeric_limits<T>::infinity();

  // Handle edge cases for p
  if (p <= T(0)) {
    // P(X = 0 | p = 0) = 1, so log(1) = 0
    // P(X = k | p = 0) = 0 for k > 0, so log(0) = -inf
    return (k == T(0)) ? T(0) : -std::numeric_limits<T>::infinity();
  }
  if (p >= T(1)) {
    // P(X = n | p = 1) = 1, so log(1) = 0
    // P(X = k | p = 1) = 0 for k < n, so log(0) = -inf
    return (k == n) ? T(0) : -std::numeric_limits<T>::infinity();
  }

  // log(PMF) = log_gamma(n+1) - log_gamma(k+1) - log_gamma(n-k+1)
  //          + k*log(p) + (n-k)*log(1-p)
  return special_functions::log_gamma(n + T(1))
       - special_functions::log_gamma(k + T(1))
       - special_functions::log_gamma(n - k + T(1))
       + k * std::log(p)
       + (n - k) * std::log(T(1) - p);
}

}  // namespace torchscience::kernel::probability
