#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

namespace torchscience::kernel::special_functions {

// Second-order backward for fresnel_s
//
// The forward pass computes: gradient * sin(pi*z^2/2)
// Let f(z) = sin(pi*z^2/2)
// f'(z) = pi*z*cos(pi*z^2/2)
//
// The backward function computes: gradient * f(z)
// We need gradients with respect to both gradient and z.
//
// d/d(gradient) [gradient * f(z)] = f(z) = sin(pi*z^2/2)
// d/dz [gradient * f(z)] = gradient * f'(z) = gradient * pi*z*cos(pi*z^2/2)
template <typename T>
std::tuple<T, T> fresnel_s_backward_backward(
    T gradient_gradient,
    T gradient,
    T z) {
  const T pi = static_cast<T>(3.14159265358979323846);
  const T pi_over_2 = static_cast<T>(1.5707963267948966);

  T arg = pi_over_2 * z * z;
  T sin_arg = std::sin(arg);
  T cos_arg = std::cos(arg);

  // Gradient w.r.t. the incoming gradient: sin(pi*z^2/2)
  T grad_gradient = gradient_gradient * sin_arg;

  // Gradient w.r.t. z: gradient * pi*z*cos(pi*z^2/2)
  T grad_z = gradient_gradient * gradient * pi * z * cos_arg;

  return {grad_gradient, grad_z};
}

// Complex version (c10::complex)
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> fresnel_s_backward_backward(
    c10::complex<T> gradient_gradient,
    c10::complex<T> gradient,
    c10::complex<T> z) {
  const T pi = static_cast<T>(3.14159265358979323846);
  const T pi_over_2 = static_cast<T>(1.5707963267948966);

  c10::complex<T> arg = pi_over_2 * z * z;
  c10::complex<T> sin_arg = c10_complex_math::sin(arg);
  c10::complex<T> cos_arg = c10_complex_math::cos(arg);

  // Gradient w.r.t. the incoming gradient: sin(pi*z^2/2)
  c10::complex<T> grad_gradient = gradient_gradient * sin_arg;

  // Gradient w.r.t. z: gradient * pi*z*cos(pi*z^2/2)
  c10::complex<T> grad_z = gradient_gradient * gradient * pi * z * cos_arg;

  return {grad_gradient, grad_z};
}

}  // namespace torchscience::kernel::special_functions
