#pragma once

#include <cmath>
#include <limits>

#include "../special_functions/log_beta.h"

namespace torchscience::kernel::probability {

// Beta log probability density function
// logpdf(x; a, b) = (a - 1) * log(x) + (b - 1) * log(1 - x) - log_beta(a, b)
//
// This computes the log PDF directly for numerical stability, rather than log(pdf).
template <typename T>
T beta_log_probability_density(T x, T a, T b) {
  if (x <= T(0) || x >= T(1)) {
    return -std::numeric_limits<T>::infinity();
  }

  return (a - T(1)) * std::log(x) + (b - T(1)) * std::log(T(1) - x)
       - special_functions::log_beta(a, b);
}

}  // namespace torchscience::kernel::probability
