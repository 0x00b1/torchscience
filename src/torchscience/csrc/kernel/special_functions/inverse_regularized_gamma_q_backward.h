#pragma once

#include <tuple>

#include "inverse_regularized_gamma_p_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> inverse_regularized_gamma_q_backward(T gradient, T a, T y) {
  // x = inverse_regularized_gamma_q(a, y) = inverse_regularized_gamma_p(a, 1 - y)
  //
  // Let p = 1 - y. Then x = P^{-1}(a, p).
  //
  // dx/da = (dP^{-1}/da)(a, p)
  // dx/dy = (dP^{-1}/dp)(a, p) * dp/dy = (dP^{-1}/dp)(a, p) * (-1)
  //
  // So we call the P backward and negate the y gradient.

  T p = T(1) - y;

  auto [grad_a, grad_p] = inverse_regularized_gamma_p_backward(gradient, a, p);

  // dy/dp = -1, so grad_y = grad_p * (-1) = -grad_p
  T grad_y = -grad_p;

  return std::make_tuple(grad_a, grad_y);
}

} // namespace torchscience::kernel::special_functions
