#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

// Real backward: d/dx Ei(x) = e^x / x
template <typename T>
T exponential_integral_ei_backward(T gradient, T x) {
  // The derivative of Ei(x) is e^x / x
  // At x = 0, this is undefined (singularity)
  if (x == T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  return gradient * std::exp(x) / x;
}

// Complex backward: d/dz Ei(z) = e^z / z
// PyTorch convention: grad * conj(d/dz Ei(z)) for Wirtinger derivatives
template <typename T>
c10::complex<T> exponential_integral_ei_backward(c10::complex<T> gradient, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // At z = 0, the derivative is undefined
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  Complex deriv = std::exp(z) / z;
  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
