#pragma once

#include <tuple>

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> gamma_sign_backward_backward(T gg_x, T grad_output, T x) {
  // Derivative is 0, so second derivative is also 0
  (void)gg_x;
  (void)grad_output;
  (void)x;
  return std::make_tuple(T(0), T(0));
}

} // namespace torchscience::kernel::special_functions
