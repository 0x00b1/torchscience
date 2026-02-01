#pragma once

#include "inverse_regularized_gamma_p.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T inverse_regularized_gamma_q(T a, T y) {
  // Inverse of the regularized upper incomplete gamma function Q(a, x)
  //
  // Since Q(a, x) = 1 - P(a, x), we have:
  //   Q^{-1}(a, y) = P^{-1}(a, 1 - y)
  //
  // Edge cases:
  //   - y = 0: Q(a, x) = 0 means x = infinity
  //   - y = 1: Q(a, x) = 1 means x = 0
  //   - a <= 0: undefined
  //   - y < 0 or y > 1: undefined

  return inverse_regularized_gamma_p(a, T(1) - y);
}

} // namespace torchscience::kernel::special_functions
