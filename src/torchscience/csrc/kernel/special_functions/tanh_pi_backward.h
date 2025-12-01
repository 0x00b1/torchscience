#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "tanh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tanh_pi_backward(T gradient, T x) {
  T tanh_pi_x = tanh_pi(x);

  return gradient * static_cast<T>(M_PI) * (T(1) - tanh_pi_x * tanh_pi_x);
}

template <typename T>
c10::complex<T> tanh_pi_backward(c10::complex<T> gradient, c10::complex<T> z) {
  c10::complex<T> tanh_pi_z = tanh_pi(z);
  c10::complex<T> one(T(1), T(0));

  return gradient * static_cast<T>(M_PI) * (one - tanh_pi_z * tanh_pi_z);
}

} // namespace torchscience::kernel::special_functions
