#pragma once

#include "inverse_regularized_incomplete_beta.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T inverse_complementary_regularized_incomplete_beta(T a, T b, T y) {
  // Inverse of the complementary regularized incomplete beta function
  //
  // Since I_c(a, b, x) = 1 - I(a, b, x), we have:
  //   I_c^{-1}(a, b, y) = I^{-1}(a, b, 1 - y)
  //
  // Edge cases:
  //   - y = 0: I_c(a, b, x) = 0 means I(a, b, x) = 1, so x = 1
  //   - y = 1: I_c(a, b, x) = 1 means I(a, b, x) = 0, so x = 0
  //   - a <= 0 or b <= 0: undefined

  return inverse_regularized_incomplete_beta(a, b, T(1) - y);
}

} // namespace torchscience::kernel::special_functions
