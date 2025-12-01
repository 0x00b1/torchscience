#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "cos_pi.h"
#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T tan_pi(T x) {
  T x_mod = std::fmod(x, T(1));

  if (x_mod < T(0)) {
    x_mod += T(1);
  }

  return std::tan(static_cast<T>(M_PI) * x_mod);
}

template <typename T>
c10::complex<T> tan_pi(c10::complex<T> z) {
  return sin_pi(z) / cos_pi(z);
}

} // namespace torchscience::kernel::special_functions
