#pragma once

#include <cmath>
#include <limits>

#include "../special_functions/log_beta.h"

namespace torchscience::kernel::probability {

// F-distribution log probability density function
// logpdf(x; dfn, dfd) = (dfn/2) * log(dfn/dfd) + (dfn/2 - 1) * log(x)
//                      - ((dfn + dfd)/2) * log(1 + dfn*x/dfd)
//                      - log_beta(dfn/2, dfd/2)
template <typename T>
T f_log_probability_density(T x, T dfn, T dfd) {
  if (x <= T(0)) {
    return -std::numeric_limits<T>::infinity();
  }

  if (std::isnan(x) || std::isnan(dfn) || std::isnan(dfd)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T half_dfn = dfn / T(2);
  T half_dfd = dfd / T(2);

  return half_dfn * std::log(dfn / dfd)
       + (half_dfn - T(1)) * std::log(x)
       - (half_dfn + half_dfd) * std::log(T(1) + dfn * x / dfd)
       - special_functions::log_beta(half_dfn, half_dfd);
}

}  // namespace torchscience::kernel::probability
