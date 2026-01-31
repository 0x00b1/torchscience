#pragma once

#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// Beta survival function (numerically stable)
// S(x; a, b) = 1 - CDF(x) = I_{1-x}(b, a)
// where I_z(p, q) is the regularized incomplete beta function
// Note: parameters are swapped and x is transformed to (1-x)
template <typename T>
T beta_survival(T x, T a, T b) {
  if (x <= T(0)) return T(1);
  if (x >= T(1)) return T(0);
  return special_functions::incomplete_beta(T(1) - x, b, a);
}

}  // namespace torchscience::kernel::probability
