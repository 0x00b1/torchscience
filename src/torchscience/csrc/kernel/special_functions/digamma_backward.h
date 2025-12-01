#pragma once

#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T digamma_backward(T gradient, T z) {
  return gradient * trigamma(z);
}

template <typename T>
c10::complex<T> digamma_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return gradient * trigamma(z);
}

} // namespace torchscience::kernel::special_functions
