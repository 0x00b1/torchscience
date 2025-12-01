#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sinh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T cosh_pi_backward(T gradient, T x) {
  return gradient * static_cast<T>(M_PI) * sinh_pi(x);
}

template <typename T>
c10::complex<T> cosh_pi_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return gradient * static_cast<T>(M_PI) * sinh_pi(z);
}

} // namespace torchscience::kernel::special_functions
