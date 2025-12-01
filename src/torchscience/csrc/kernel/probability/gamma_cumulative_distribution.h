#pragma once

#include "../special_functions/regularized_gamma_p.h"

namespace torchscience::kernel::probability {

// Gamma CDF: F(x; shape, scale) = P(shape, x/scale)
// where P is the regularized lower incomplete gamma function
template <typename T>
T gamma_cumulative_distribution(T x, T shape, T scale) {
  if (x <= T(0)) return T(0);
  return special_functions::regularized_gamma_p(shape, x / scale);
}

}  // namespace torchscience::kernel::probability
