#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "cos_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tan_pi_backward(T gradient, T x) {
  T cos_pi_x = cos_pi(x);

  return gradient * static_cast<T>(M_PI) / (cos_pi_x * cos_pi_x);
}

template <typename T>
c10::complex<T> tan_pi_backward(c10::complex<T> gradient, c10::complex<T> z) {
  c10::complex<T> cos_pi_z = cos_pi(z);

  return gradient * static_cast<T>(M_PI) / (cos_pi_z * cos_pi_z);
}

} // namespace torchscience::kernel::special_functions
