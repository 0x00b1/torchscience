#pragma once

#include <cmath>
#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// Binomial survival function: S(k; n, p) = P(X > k) = I_p(k+1, n-k)
// where I_x(a, b) is the regularized incomplete beta function
// This gives the probability that a Binomial(n, p) random variable exceeds k.
template <typename T>
T binomial_survival(T k, T n, T p) {
  // Floor k for non-integer inputs
  k = std::floor(k);

  // Boundary cases
  // k < 0: all probability mass is > k, so survival = 1
  if (k < T(0)) return T(1);

  // k >= n: no probability mass is > k, so survival = 0
  if (k >= n) return T(0);

  // Edge cases for p
  // p <= 0: deterministic 0 successes, survival = 1 if k < 0 (handled above), else 0
  if (p <= T(0)) return T(0);

  // p >= 1: deterministic n successes, survival = 1 if k < n, else 0
  if (p >= T(1)) return T(1);

  // S(k) = P(X > k) = 1 - P(X <= k) = 1 - CDF(k)
  // CDF(k) = I_{1-p}(n-k, k+1) (regularized incomplete beta)
  // S(k) = 1 - I_{1-p}(n-k, k+1) = I_p(k+1, n-k)
  // Using the identity: 1 - I_x(a, b) = I_{1-x}(b, a)
  return special_functions::incomplete_beta(p, k + T(1), n - k);
}

}  // namespace torchscience::kernel::probability
