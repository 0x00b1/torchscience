#pragma once

#include "../special_functions/incomplete_beta_backward.h"

namespace torchscience::kernel::probability {

// Beta CDF backward is the incomplete beta backward
template <typename T>
std::tuple<T, T, T> beta_cumulative_distribution_backward(T gradient, T x, T a, T b) {
  return special_functions::incomplete_beta_backward(gradient, x, a, b);
}

}  // namespace torchscience::kernel::probability
