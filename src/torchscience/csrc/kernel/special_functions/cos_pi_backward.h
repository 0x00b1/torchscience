#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T cos_pi_backward(T gradient, T x) {
  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  return -gradient * static_cast<T>(M_PI) * std::sin(static_cast<T>(M_PI) * x_mod);
}

template <typename T>
c10::complex<T> cos_pi_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return -gradient * static_cast<T>(M_PI) * sin_pi(z);
}

} // namespace torchscience::kernel::special_functions
