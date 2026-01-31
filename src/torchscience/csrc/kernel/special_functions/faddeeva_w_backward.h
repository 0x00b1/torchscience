#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "faddeeva_w.h"

namespace torchscience::kernel::special_functions {

// Derivative of the Faddeeva function:
// d/dz w(z) = -2z*w(z) + 2i/sqrt(pi)
//
// This follows from w(z) = exp(-z^2) * erfc(-iz) and the chain rule.

namespace detail {

template <typename T>
struct faddeeva_backward_constants {
  static constexpr T two_over_sqrt_pi = T(1.1283791670955125738961589031215451716);  // 2/sqrt(pi)
};

}  // namespace detail

// Complex backward: PyTorch expects grad * conj(d/dz w(z)) for holomorphic functions
// This handles both y.real.backward() (grad=1+0j) and y.imag.backward() (grad=0+1j)
template <typename T>
c10::complex<T> faddeeva_w_backward(c10::complex<T> gradient, c10::complex<T> z) {
  c10::complex<T> w_z = faddeeva_w(z);

  // d/dz w(z) = -2*z*w(z) + 2i/sqrt(pi)
  c10::complex<T> two_i_sqrt_pi(T(0), detail::faddeeva_backward_constants<T>::two_over_sqrt_pi);
  c10::complex<T> deriv = c10::complex<T>(T(-2), T(0)) * z * w_z + two_i_sqrt_pi;

  // PyTorch convention: grad * conj(df/dz) for complex autograd
  return gradient * std::conj(deriv);
}

}  // namespace torchscience::kernel::special_functions
