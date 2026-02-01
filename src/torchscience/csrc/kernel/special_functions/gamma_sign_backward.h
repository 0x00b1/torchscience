#pragma once

namespace torchscience::kernel::special_functions {

template <typename T>
T gamma_sign_backward(T gradient, T x) {
  // gamma_sign is piecewise constant, derivative is 0
  (void)gradient;
  (void)x;
  return T(0);
}

} // namespace torchscience::kernel::special_functions
