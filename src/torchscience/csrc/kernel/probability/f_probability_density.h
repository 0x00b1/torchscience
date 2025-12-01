#pragma once

#include <cmath>
#include <limits>
#include "../special_functions/log_beta.h"

namespace torchscience::kernel::probability {

// F-distribution PDF:
// pdf(x; d1, d2) = sqrt((d1*x)^d1 * d2^d2 / (d1*x + d2)^(d1+d2)) / (x * B(d1/2, d2/2))
// In log form for numerical stability:
// log(pdf) = (d1/2)*log(d1) + (d2/2)*log(d2) + ((d1-2)/2)*log(x)
//            - ((d1+d2)/2)*log(d1*x + d2) - log_beta(d1/2, d2/2)
template <typename T>
T f_probability_density(T x, T d1, T d2) {
  if (x < T(0)) return T(0);
  if (std::isnan(x) || std::isnan(d1) || std::isnan(d2)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // Handle x = 0 case
  if (x == T(0)) {
    if (d1 < T(2)) return std::numeric_limits<T>::infinity();
    if (d1 == T(2)) {
      // pdf(0) = 1 when d1 = 2
      return T(1);
    }
    return T(0);  // d1 > 2
  }

  T half_d1 = d1 / T(2);
  T half_d2 = d2 / T(2);

  T log_probability_density = half_d1 * std::log(d1) + half_d2 * std::log(d2)
            + (half_d1 - T(1)) * std::log(x)
            - (half_d1 + half_d2) * std::log(d1 * x + d2)
            - special_functions::log_beta(half_d1, half_d2);

  return std::exp(log_probability_density);
}

}  // namespace torchscience::kernel::probability
