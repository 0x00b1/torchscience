#pragma once

#include <cmath>
#include <tuple>
#include "regularized_gamma_p_backward.h"

namespace torchscience::kernel::special_functions {

// Gradient of Q(a, x) = 1 - P(a, x)
// dQ/da = -dP/da, dQ/dx = -dP/dx

template <typename T>
std::tuple<T, T> regularized_gamma_q_backward(T grad_output, T a, T x) {
  auto [grad_a_p, grad_x_p] = regularized_gamma_p_backward(grad_output, a, x);
  return {-grad_a_p, -grad_x_p};
}

}  // namespace torchscience::kernel::special_functions
