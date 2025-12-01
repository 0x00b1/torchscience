#pragma once

#include <cmath>

namespace torchscience::kernel::probability {

// Binomial probability mass function: P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
// Computed in log space for numerical stability
template <typename T>
T binomial_probability_mass(T k, T n, T p) {
  k = std::floor(k);

  // Boundary cases
  if (k < T(0) || k > n) return T(0);

  // Handle edge cases for p
  if (p <= T(0)) {
    return (k == T(0)) ? T(1) : T(0);
  }
  if (p >= T(1)) {
    return (k == n) ? T(1) : T(0);
  }

  // PMF = C(n, k) * p^k * (1-p)^(n-k)
  // log(PMF) = lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1) + k*log(p) + (n-k)*log(1-p)
  T log_pmf = std::lgamma(n + T(1)) - std::lgamma(k + T(1)) - std::lgamma(n - k + T(1))
            + k * std::log(p) + (n - k) * std::log(T(1) - p);

  return std::exp(log_pmf);
}

}  // namespace torchscience::kernel::probability
