#pragma once

#include <cmath>
#include "../special_functions/regularized_gamma_p.h"

namespace torchscience::kernel::probability {

// Poisson survival function: S(k; rate) = P(X > k) = P(k+1, rate)
// where P is the lower regularized incomplete gamma function
// This gives the probability that a Poisson(rate) random variable exceeds k.
template <typename T>
T poisson_survival(T k, T rate) {
  k = std::floor(k);

  // Boundary cases
  // k < 0: all probability mass is > k, so survival = 1
  if (k < T(0)) return T(1);

  // rate <= 0: no valid distribution, return 0
  if (rate <= T(0)) return T(0);

  // S(k) = P(X > k) = 1 - P(X <= k) = 1 - Q(k+1, rate) = P(k+1, rate)
  // where P is the lower regularized incomplete gamma function
  return special_functions::regularized_gamma_p(k + T(1), rate);
}

}  // namespace torchscience::kernel::probability
