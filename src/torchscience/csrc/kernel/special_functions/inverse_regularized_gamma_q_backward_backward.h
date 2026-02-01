#pragma once

#include <tuple>

#include "inverse_regularized_gamma_p_backward_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> inverse_regularized_gamma_q_backward_backward(
    T gg_a, T gg_y, T grad_output, T a, T y) {
  // Second-order derivatives for inverse_regularized_gamma_q
  //
  // x = Q^{-1}(a, y) = P^{-1}(a, 1 - y)
  //
  // Let p = 1 - y. The relationship between Q and P gradients:
  //   dx/da is the same (doesn't depend on whether we use p or y)
  //   dx/dy = -dx/dp (chain rule with dp/dy = -1)
  //
  // For second derivatives:
  //   d²x/da² is the same
  //   d²x/dy² = d²x/dp² (double negation cancels)
  //   d²x/dady = -d²x/dadp (chain rule once)
  //
  // The backward_backward function computes:
  //   grad_grad_output = gg_a * dx/da + gg_y * dx/dy
  //   grad_a = grad_output * (gg_a * d²x/da² + gg_y * d²x/dady)
  //   grad_y = grad_output * (gg_a * d²x/dady + gg_y * d²x/dy²)
  //
  // Converting from P to Q:
  //   For P: dx/dp and d²x/dadp
  //   For Q: dx/dy = -dx/dp and d²x/dady = -d²x/dadp
  //
  // So we transform gg_y to -gg_y when calling P's backward_backward,
  // then negate grad_y in the output.

  T p = T(1) - y;

  // Transform: gg_y for Q corresponds to -gg_p for P
  // because dx/dy = -dx/dp, so gg_y * dx/dy = gg_y * (-dx/dp) = (-gg_y) * dx/dp
  T gg_p = -gg_y;

  auto [grad_grad_output, grad_a_out, grad_p_out] =
      inverse_regularized_gamma_p_backward_backward(gg_a, gg_p, grad_output, a, p);

  // Transform back: grad_y = -grad_p because dy/dp = -1
  T grad_y_out = -grad_p_out;

  return std::make_tuple(grad_grad_output, grad_a_out, grad_y_out);
}

} // namespace torchscience::kernel::special_functions
