#pragma once

#include "tetragamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T trigamma_backward(
  T gradient,
  T z
) {
  return gradient * tetragamma(z);
}

template <typename T>
c10::complex<T> trigamma_backward(
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return gradient * tetragamma(z);
}

} // namespace torchscience::kernel::special_functions
