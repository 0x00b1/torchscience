#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::special_functions {

// Real backward_backward
// d/dx Si(x) = sin(x) / x
// d^2/dx^2 Si(x) = (x * cos(x) - sin(x)) / x^2
// At x = 0, use Taylor expansion: Si''(0) = 0
// (since sinc(x) = 1 - x^2/6 + x^4/120 - ..., so sinc'(0) = 0)
template <typename T>
std::tuple<T, T> sine_integral_si_backward_backward(
  T gradient_gradient,
  T gradient,
  T x
) {
  T first_deriv;   // d/dx Si(x) = sin(x)/x
  T second_deriv;  // d^2/dx^2 Si(x) = (x*cos(x) - sin(x))/x^2

  if (x == T(0)) {
    // Removable singularities using limits
    // lim_{x->0} sin(x)/x = 1
    first_deriv = T(1);
    // Using Taylor: sinc(x) = 1 - x^2/6 + x^4/120 - ...
    // sinc'(x) = -x/3 + x^3/30 - ...
    // So sinc'(0) = 0, meaning Si''(0) = 0
    second_deriv = T(0);
  } else {
    T sin_x = std::sin(x);
    T cos_x = std::cos(x);
    first_deriv = sin_x / x;
    second_deriv = (x * cos_x - sin_x) / (x * x);
  }

  return {
    gradient_gradient * first_deriv,         // grad w.r.t. gradient
    gradient_gradient * gradient * second_deriv  // grad w.r.t. x
  };
}

// Complex backward_backward
// d/dz Si(z) = sin(z) / z
// d^2/dz^2 Si(z) = (z * cos(z) - sin(z)) / z^2
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> sine_integral_si_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  using Complex = c10::complex<T>;

  Complex first_deriv;   // d/dz Si(z)
  Complex second_deriv;  // d^2/dz^2 Si(z)

  if (z.real() == T(0) && z.imag() == T(0)) {
    // Removable singularities using limits
    first_deriv = Complex(T(1), T(0));
    second_deriv = Complex(T(0), T(0));
  } else {
    Complex sin_z = std::sin(z);
    Complex cos_z = std::cos(z);
    first_deriv = sin_z / z;
    second_deriv = (z * cos_z - sin_z) / (z * z);
  }

  // PyTorch convention: grad * conj(derivative) for holomorphic functions
  Complex grad_gradient = gradient_gradient * std::conj(first_deriv);
  Complex grad_z = gradient_gradient * gradient * std::conj(second_deriv);

  return {grad_gradient, grad_z};
}

} // namespace torchscience::kernel::special_functions
