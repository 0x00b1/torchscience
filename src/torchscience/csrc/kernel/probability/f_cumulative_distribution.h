#pragma once

#include <cmath>
#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// F-distribution CDF: I_{x'}(d1/2, d2/2) where x' = d1*f / (d1*f + d2)
template <typename T>
T f_cumulative_distribution(T f, T d1, T d2) {
  if (f <= T(0)) return T(0);
  if (std::isnan(f) || std::isnan(d1) || std::isnan(d2)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T x_prime = d1 * f / (d1 * f + d2);
  return special_functions::incomplete_beta(x_prime, d1 / T(2), d2 / T(2));
}

}  // namespace torchscience::kernel::probability
