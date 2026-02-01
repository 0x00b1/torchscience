#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "digamma.h"
#include "pochhammer.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> pochhammer_backward(T gradient, T z, T m) {
  // (z)_m = Gamma(z+m) / Gamma(z)
  // d/dz (z)_m = (z)_m * (psi(z+m) - psi(z))
  // d/dm (z)_m = (z)_m * psi(z+m)

  T poch = pochhammer(z, m);
  T psi_zm = digamma(z + m);
  T psi_z = digamma(z);

  T grad_z = gradient * poch * (psi_zm - psi_z);
  T grad_m = gradient * poch * psi_zm;

  return std::make_tuple(grad_z, grad_m);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> pochhammer_backward(
    c10::complex<T> gradient, c10::complex<T> z, c10::complex<T> m) {
  c10::complex<T> poch = pochhammer(z, m);
  c10::complex<T> psi_zm = digamma(z + m);
  c10::complex<T> psi_z = digamma(z);

  c10::complex<T> grad_z = gradient * poch * (psi_zm - psi_z);
  c10::complex<T> grad_m = gradient * poch * psi_zm;

  return std::make_tuple(grad_z, grad_m);
}

} // namespace torchscience::kernel::special_functions
