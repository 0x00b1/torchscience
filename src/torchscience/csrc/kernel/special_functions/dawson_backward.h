#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "dawson.h"

namespace torchscience::kernel::special_functions {

// Gradient of Dawson's integral
// d/dz D(z) = 1 - 2z * D(z)
//
// Derivation:
// D(z) = exp(-z^2) * integral from 0 to z of exp(t^2) dt
// d/dz D(z) = -2z * exp(-z^2) * integral + exp(-z^2) * exp(z^2)
//           = -2z * D(z) + 1
//           = 1 - 2z * D(z)
template <typename T>
T dawson_backward(T gradient, T z) {
  T D_z = dawson(z);
  T dD_dz = T(1) - T(2) * z * D_z;
  return gradient * dD_dz;
}

// Complex version
template <typename T>
c10::complex<T> dawson_backward(c10::complex<T> gradient, c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> D_z = dawson(z);
  c10::complex<T> dD_dz = one - two * z * D_z;
  return gradient * std::conj(dD_dz);
}

}  // namespace torchscience::kernel::special_functions
