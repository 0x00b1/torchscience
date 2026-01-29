#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Real backward: d/dx Ci(x) = cos(x) / x
// At x = 0: undefined (Ci has a singularity there)
// At x > 0: the derivative is well-defined
template <typename T>
T cosine_integral_ci_backward(T gradient, T x) {
  // Ci is only defined for x > 0, and x = 0 is a singularity
  if (x <= T(0)) {
    return std::numeric_limits<T>::quiet_NaN();
  }

  T deriv = std::cos(x) / x;
  return gradient * deriv;
}

// Complex backward: d/dz Ci(z) = cos(z) / z
// PyTorch convention: grad * conj(d/dz Ci(z)) for Wirtinger derivatives
template <typename T>
c10::complex<T> cosine_integral_ci_backward(c10::complex<T> gradient, c10::complex<T> z) {
  using Complex = c10::complex<T>;

  // z = 0: singularity
  if (z.real() == T(0) && z.imag() == T(0)) {
    return Complex(std::numeric_limits<T>::quiet_NaN(),
                   std::numeric_limits<T>::quiet_NaN());
  }

  Complex deriv = std::cos(z) / z;
  return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
