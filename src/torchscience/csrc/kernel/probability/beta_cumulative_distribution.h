#pragma once

#include "../special_functions/incomplete_beta.h"

namespace torchscience::kernel::probability {

// Beta CDF is the regularized incomplete beta function: I_x(a, b)
// F(x; a, b) = I_x(a, b) = B(x; a, b) / B(a, b)
template <typename T>
T beta_cumulative_distribution(T x, T a, T b) {
  return special_functions::incomplete_beta(x, a, b);
}

}  // namespace torchscience::kernel::probability
