#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::special_functions {

// Real backward_backward
// d/dx E_1(x) = -e^{-x} / x
// d^2/dx^2 E_1(x) = e^{-x} * (x + 1) / x^2
template <typename T>
std::tuple<T, T> exponential_integral_e_1_backward_backward(
  T gradient_gradient,
  T gradient,
  T x
) {
  if (x == T(0)) {
    return {
      std::numeric_limits<T>::quiet_NaN(),
      std::numeric_limits<T>::quiet_NaN()
    };
  }

  T exp_neg_x = std::exp(-x);
  T first_deriv = -exp_neg_x / x;  // d/dx E_1(x)
  T second_deriv = exp_neg_x * (x + T(1)) / (x * x);  // d^2/dx^2 E_1(x)

  return {
    gradient_gradient * first_deriv,        // grad w.r.t. gradient
    gradient_gradient * gradient * second_deriv  // grad w.r.t. x
  };
}

// Complex backward_backward
// d/dz E_1(z) = -e^{-z} / z
// d^2/dz^2 E_1(z) = e^{-z} * (z + 1) / z^2
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> exponential_integral_e_1_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  using Complex = c10::complex<T>;

  if (z.real() == T(0) && z.imag() == T(0)) {
    return {
      Complex(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()),
      Complex(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN())
    };
  }

  Complex exp_neg_z = std::exp(-z);
  Complex first_deriv = -exp_neg_z / z;  // d/dz E_1(z)
  Complex second_deriv = exp_neg_z * (z + Complex(T(1), T(0))) / (z * z);  // d^2/dz^2 E_1(z)

  // PyTorch convention: grad * conj(derivative) for holomorphic functions
  Complex grad_gradient = gradient_gradient * std::conj(first_deriv);
  Complex grad_z = gradient_gradient * gradient * std::conj(second_deriv);

  return {grad_gradient, grad_z};
}

} // namespace torchscience::kernel::special_functions
