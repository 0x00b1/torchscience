#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::special_functions {

// Real backward_backward
// d/dx Ein(x) = (1 - e^(-x)) / x
// d^2/dx^2 Ein(x) = (e^(-x)(x + 1) - 1) / x^2
// At x = 0, use Taylor expansion: Ein''(0) = -1/2
template <typename T>
std::tuple<T, T> exponential_integral_ein_backward_backward(
  T gradient_gradient,
  T gradient,
  T x
) {
  T first_deriv;   // d/dx Ein(x)
  T second_deriv;  // d^2/dx^2 Ein(x)

  if (x == T(0)) {
    // Removable singularities using limits
    // lim_{x->0} (1 - e^(-x))/x = 1
    first_deriv = T(1);
    // Using Taylor: Ein(x) = x - x^2/4 + x^3/18 - ...
    // Ein'(x) = 1 - x/2 + x^2/6 - ...
    // Ein''(x) = -1/2 + x/3 - ...
    // So Ein''(0) = -1/2
    second_deriv = T(-0.5);
  } else {
    T exp_neg_x = std::exp(-x);
    first_deriv = (T(1) - exp_neg_x) / x;
    second_deriv = (exp_neg_x * (x + T(1)) - T(1)) / (x * x);
  }

  return {
    gradient_gradient * first_deriv,         // grad w.r.t. gradient
    gradient_gradient * gradient * second_deriv  // grad w.r.t. x
  };
}

// Complex backward_backward
// d/dz Ein(z) = (1 - e^(-z)) / z
// d^2/dz^2 Ein(z) = (e^(-z)(z + 1) - 1) / z^2
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> exponential_integral_ein_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  using Complex = c10::complex<T>;

  Complex first_deriv;   // d/dz Ein(z)
  Complex second_deriv;  // d^2/dz^2 Ein(z)

  if (z.real() == T(0) && z.imag() == T(0)) {
    // Removable singularities using limits
    first_deriv = Complex(T(1), T(0));
    second_deriv = Complex(T(-0.5), T(0));
  } else {
    Complex exp_neg_z = std::exp(-z);
    first_deriv = (Complex(T(1), T(0)) - exp_neg_z) / z;
    second_deriv = (exp_neg_z * (z + Complex(T(1), T(0))) - Complex(T(1), T(0))) / (z * z);
  }

  // PyTorch convention: grad * conj(derivative) for holomorphic functions
  Complex grad_gradient = gradient_gradient * std::conj(first_deriv);
  Complex grad_z = gradient_gradient * gradient * std::conj(second_deriv);

  return {grad_gradient, grad_z};
}

} // namespace torchscience::kernel::special_functions
