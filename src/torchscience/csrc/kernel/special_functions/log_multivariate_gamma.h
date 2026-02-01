#pragma once

#include <cmath>

#include "log_gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T log_multivariate_gamma(T a, int64_t d) {
  // log(Gamma_d(a)) = d*(d-1)/4 * log(pi) + sum_{j=1}^{d} log_gamma(a + (1-j)/2)
  // Valid for a > (d-1)/2

  T result = static_cast<T>(d * (d - 1)) / T(4) * std::log(static_cast<T>(M_PI));

  for (int64_t j = 1; j <= d; ++j) {
    result += log_gamma(a + (T(1) - static_cast<T>(j)) / T(2));
  }

  return result;
}

} // namespace torchscience::kernel::special_functions
