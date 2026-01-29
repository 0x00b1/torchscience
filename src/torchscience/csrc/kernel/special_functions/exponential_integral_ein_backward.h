#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

// Real backward: d/dx Ein(x) = (1 - e^(-x)) / x
// At x = 0, the derivative has a removable singularity: lim_{x->0} (1 - e^(-x))/x = 1
template <typename T>
T exponential_integral_ein_backward(T gradient, T x) {
  T deriv;

  if (x == T(0)) {
    // Removable singularity: lim_{x->0} (1 - e^(-x))/x = 1
    deriv = T(1);
  } else {
    deriv = (T(1) - std::exp(-x)) / x;
  }

  return gradient * deriv;
}

// Complex backward: d/dz Ein(z) = (1 - e^(-z)) / z
// PyTorch convention: grad * conj(d/dz Ein(z)) for Wirtinger derivatives
template <typename T>
c10::complex<T> exponential_integral_ein_backward(c10::complex<T> gradient, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  Complex deriv;

  if (z.real() == T(0) && z.imag() == T(0)) {
    // Removable singularity: lim_{z->0} (1 - e^(-z))/z = 1
    deriv = Complex(T(1), T(0));
  } else {
    deriv = (Complex(T(1), T(0)) - std::exp(-z)) / z;
  }

  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
