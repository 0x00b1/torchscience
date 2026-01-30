#pragma once

#include "../special_functions/regularized_gamma_q.h"

namespace torchscience::kernel::probability {

// Gamma survival function (numerically stable)
// S(x; shape, scale) = 1 - CDF(x) = Q(shape, x/scale)
// where Q is the regularized upper incomplete gamma function
template <typename T>
T gamma_survival(T x, T shape, T scale) {
  if (x <= T(0)) return T(1);
  return special_functions::regularized_gamma_q(shape, x / scale);
}

}  // namespace torchscience::kernel::probability
