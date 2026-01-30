#pragma once

#include <cmath>
#include <limits>

#include "../special_functions/log_gamma.h"

namespace torchscience::kernel::probability {

// Gamma log probability density function
// logpdf(x; shape, scale) = (shape - 1) * log(x) - x/scale - log_gamma(shape) - shape * log(scale)
//
// This computes the log PDF directly for numerical stability, rather than log(pdf).
template <typename T>
T gamma_log_probability_density(T x, T shape, T scale) {
  if (x <= T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  return (shape - T(1)) * std::log(x) - x / scale
       - special_functions::log_gamma(shape) - shape * std::log(scale);
}

}  // namespace torchscience::kernel::probability
