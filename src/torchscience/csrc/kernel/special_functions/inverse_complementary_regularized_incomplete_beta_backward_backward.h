#pragma once

#include <tuple>

#include "inverse_regularized_incomplete_beta_backward_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> inverse_complementary_regularized_incomplete_beta_backward_backward(
    T gg_a, T gg_b, T gg_y, T grad_output, T a, T b, T y) {
  // Second-order derivatives for inverse_complementary_regularized_incomplete_beta
  //
  // x = I_c^{-1}(a, b, y) = I^{-1}(a, b, 1 - y)
  //
  // Let p = 1 - y. The relationship between complementary and regular gradients:
  //   dx/da is the same (doesn't depend on whether we use p or y)
  //   dx/db is the same
  //   dx/dy = -dx/dp (chain rule with dp/dy = -1)
  //
  // For second derivatives:
  //   d^2x/da^2, d^2x/db^2, d^2x/dadb are the same
  //   d^2x/dy^2 = d^2x/dp^2 (double negation cancels)
  //   d^2x/dady = -d^2x/dadp
  //   d^2x/dbdy = -d^2x/dbdp
  //
  // We transform gg_y to -gg_p when calling the regular backward_backward,
  // then negate grad_y in the output.

  T p = T(1) - y;

  // Transform: gg_y for complementary corresponds to -gg_p for regular
  T gg_p = -gg_y;

  auto [grad_grad_output, grad_a_out, grad_b_out, grad_p_out] =
      inverse_regularized_incomplete_beta_backward_backward(gg_a, gg_b, gg_p, grad_output, a, b, p);

  // Transform back: grad_y = -grad_p because dy/dp = -1
  T grad_y_out = -grad_p_out;

  return std::make_tuple(grad_grad_output, grad_a_out, grad_b_out, grad_y_out);
}

} // namespace torchscience::kernel::special_functions
