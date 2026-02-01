#pragma once

#include <c10/util/complex.h>

#include "digamma.h"
#include "reciprocal_gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T reciprocal_gamma_backward(T gradient, T z) {
  // d/dz [1/Gamma(z)] = -psi(z) / Gamma(z) = -psi(z) * reciprocal_gamma(z)
  return gradient * (-digamma(z) * reciprocal_gamma(z));
}

template <typename T>
c10::complex<T> reciprocal_gamma_backward(c10::complex<T> gradient, c10::complex<T> z) {
  return gradient * (-digamma(z) * reciprocal_gamma(z));
}

} // namespace torchscience::kernel::special_functions
