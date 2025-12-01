#pragma once

#include <c10/util/complex.h>

#include "digamma.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

// Real backward: d/dz Gamma(z) = Gamma(z) * psi(z)
template <typename T>
T gamma_backward(T gradient, T z) {
  return gradient * gamma(z) * digamma(z);
}

// Complex backward: PyTorch expects grad * conj(d/dz Gamma(z)) for holomorphic functions
// This handles both y.real.backward() (grad=1+0j) and y.imag.backward() (grad=0+1j)
template <typename T>
c10::complex<T> gamma_backward(c10::complex<T> gradient, c10::complex<T> z) {
  c10::complex<T> deriv = gamma(z) * digamma(z);
  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
