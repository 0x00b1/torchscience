#pragma once

#include <c10/util/complex.h>
#include <cmath>

namespace torchscience::kernel::special_functions {

// Real backward: d/dx Si(x) = sin(x) / x = sinc(x)
// At x = 0, the derivative has a removable singularity: lim_{x->0} sin(x)/x = 1
template <typename T>
T sine_integral_si_backward(T gradient, T x) {
  T deriv;

  if (x == T(0)) {
    // Removable singularity: lim_{x->0} sin(x)/x = 1
    deriv = T(1);
  } else {
    deriv = std::sin(x) / x;
  }

  return gradient * deriv;
}

// Complex backward: d/dz Si(z) = sin(z) / z
// PyTorch convention: grad * conj(d/dz Si(z)) for Wirtinger derivatives
template <typename T>
c10::complex<T> sine_integral_si_backward(c10::complex<T> gradient, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  Complex deriv;

  if (z.real() == T(0) && z.imag() == T(0)) {
    // Removable singularity: lim_{z->0} sin(z)/z = 1
    deriv = Complex(T(1), T(0));
  } else {
    deriv = std::sin(z) / z;
  }

  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
