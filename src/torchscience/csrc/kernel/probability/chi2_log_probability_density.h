#pragma once

#include <cmath>
#include <limits>

#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::probability {

// Chi-squared log probability density function
// logpdf(x; df) = (df/2 - 1) * log(x) - x/2 - (df/2) * log(2) - log_gamma(df/2)
//
// This computes the log PDF directly for numerical stability, rather than log(pdf).
template <typename T>
T chi2_log_probability_density(T x, T df) {
  if (x <= T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  const T log_2 = T(0.6931471805599453);  // log(2)
  T k_half = df / T(2);

  return (k_half - T(1)) * std::log(x) - x / T(2)
       - k_half * log_2 - special_functions::log_gamma(k_half);
}

}  // namespace torchscience::kernel::probability
