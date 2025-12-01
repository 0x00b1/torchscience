#pragma once

#include <cmath>
#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// F-distribution SF (survival function): 1 - CDF
// SF = 1 - I_{x'}(d1/2, d2/2) = I_{1-x'}(d2/2, d1/2)
// where x' = d1*f / (d1*f + d2)
// Using symmetry: I_x(a,b) = 1 - I_{1-x}(b,a)
template <typename T>
T f_survival(T f, T d1, T d2) {
  if (f <= T(0)) return T(1);
  if (std::isnan(f) || std::isnan(d1) || std::isnan(d2)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Use complementary formula for better numerical stability
  // 1 - x' = d2 / (d1*f + d2)
  T one_minus_x_prime = d2 / (d1 * f + d2);
  return special_functions::incomplete_beta(one_minus_x_prime, d2 / T(2), d1 / T(2));
}

}  // namespace torchscience::kernel::probability
