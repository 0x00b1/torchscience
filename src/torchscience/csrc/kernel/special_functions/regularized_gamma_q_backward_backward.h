#pragma once

#include <cmath>
#include <tuple>
#include "regularized_gamma_p_backward_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order gradients for regularized_gamma_q
// Since Q = 1 - P, the second derivatives have the same magnitude but signs may differ

template <typename T>
std::tuple<T, T, T> regularized_gamma_q_backward_backward(
    T grad_grad_a, T grad_grad_x, T grad, T a, T x) {

  auto [gg_output_p, grad_a_p, grad_x_p] =
      regularized_gamma_p_backward_backward(grad_grad_a, grad_grad_x, grad, a, x);

  // The gradient of the output w.r.t. grad_grad is negated because Q = 1 - P
  // But the second derivatives of a and x remain the same in magnitude
  return {-gg_output_p, -grad_a_p, -grad_x_p};
}

}  // namespace torchscience::kernel::special_functions
