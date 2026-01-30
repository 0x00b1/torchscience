#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

namespace torchscience::kernel::special_functions {

// Real backward_backward
// d/dx Ci(x) = cos(x) / x
// d^2/dx^2 Ci(x) = d/dx [cos(x)/x] = (-x*sin(x) - cos(x)) / x^2
//                                  = -sin(x)/x - cos(x)/x^2
template <typename T>
std::tuple<T, T> cosine_integral_ci_backward_backward(
  T gradient_gradient,
  T gradient,
  T x
) {
  // Ci is only defined for x > 0, and x = 0 is a singularity
  if (x <= T(0)) {
    return {std::numeric_limits<T>::quiet_NaN(),
            std::numeric_limits<T>::quiet_NaN()};
  }

  T sin_x = std::sin(x);
  T cos_x = std::cos(x);
  T first_deriv = cos_x / x;
  T second_deriv = (-x * sin_x - cos_x) / (x * x);

  return {
    gradient_gradient * first_deriv,              // grad w.r.t. gradient
    gradient_gradient * gradient * second_deriv   // grad w.r.t. x
  };
}

// Complex backward_backward
// d/dz Ci(z) = cos(z) / z
// d^2/dz^2 Ci(z) = (-z*sin(z) - cos(z)) / z^2
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> cosine_integral_ci_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  using Complex = c10::complex<T>;

  // z = 0: singularity
  if (z.real() == T(0) && z.imag() == T(0)) {
    return {Complex(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()),
            Complex(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN())};
  }

  Complex sin_z = std::sin(z);
  Complex cos_z = std::cos(z);
  Complex first_deriv = cos_z / z;
  Complex second_deriv = (-z * sin_z - cos_z) / (z * z);

  // PyTorch convention: grad * conj(derivative) for holomorphic functions
  Complex grad_gradient = gradient_gradient * std::conj(first_deriv);
  Complex grad_z = gradient_gradient * gradient * std::conj(second_deriv);

  return {grad_gradient, grad_z};
}

} // namespace torchscience::kernel::special_functions
