#pragma once

#include "digamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T log_gamma_backward(T gradient, T z) {
  return gradient * digamma(z);
}

template <typename T>
c10::complex<T> log_gamma_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return gradient * digamma(z);
}

} // namespace torchscience::kernel::special_functions
