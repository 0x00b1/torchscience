#pragma once

#include <tuple>

#include "inverse_regularized_incomplete_beta_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> inverse_complementary_regularized_incomplete_beta_backward(
    T gradient, T a, T b, T y) {
  // x = inverse_complementary_regularized_incomplete_beta(a, b, y)
  //   = inverse_regularized_incomplete_beta(a, b, 1 - y)
  //
  // Let p = 1 - y. Then x = I^{-1}(a, b, p).
  //
  // dx/da = (dI^{-1}/da)(a, b, p)
  // dx/db = (dI^{-1}/db)(a, b, p)
  // dx/dy = (dI^{-1}/dp)(a, b, p) * dp/dy = (dI^{-1}/dp)(a, b, p) * (-1)
  //
  // So we call the forward's backward and negate the y gradient.

  T p = T(1) - y;

  auto [grad_a, grad_b, grad_p] = inverse_regularized_incomplete_beta_backward(gradient, a, b, p);

  // dy/dp = -1, so grad_y = grad_p * (-1) = -grad_p
  T grad_y = -grad_p;

  return std::make_tuple(grad_a, grad_b, grad_y);
}

} // namespace torchscience::kernel::special_functions
