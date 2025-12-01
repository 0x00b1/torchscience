#pragma once

#include <cmath>
#include <limits>

#include "regularized_gamma_p.h"

namespace torchscience::kernel::special_functions {

// Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x)
// Q(a, x) = Gamma(a, x) / Gamma(a) where Gamma(a, x) is the upper incomplete gamma

template <typename T>
T regularized_gamma_q(T a, T x) {
  if (x < T(0) || a <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  if (x == T(0)) return T(1);

  // Q(a, x) = 1 - P(a, x)
  return T(1) - regularized_gamma_p(a, x);
}

}  // namespace torchscience::kernel::special_functions
