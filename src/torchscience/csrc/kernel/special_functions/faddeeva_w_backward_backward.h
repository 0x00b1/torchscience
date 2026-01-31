#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "faddeeva_w.h"

namespace torchscience::kernel::special_functions {

// Second derivative of the Faddeeva function:
// d/dz w(z) = -2z*w(z) + 2i/sqrt(pi)
// d^2/dz^2 w(z) = -2*w(z) + -2z*(-2z*w(z) + 2i/sqrt(pi))
//               = -2*w(z) + 4z^2*w(z) - 4iz/sqrt(pi)
//               = (4z^2 - 2)*w(z) - 4iz/sqrt(pi)

namespace detail {

template <typename T>
struct faddeeva_second_deriv_constants {
  static constexpr T four_over_sqrt_pi = T(2.2567583341910251477923178062430903432);  // 4/sqrt(pi)
  static constexpr T two_over_sqrt_pi = T(1.1283791670955125738961589031215451716);   // 2/sqrt(pi)
};

}  // namespace detail

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> faddeeva_w_backward_backward(
    c10::complex<T> gradient_gradient,
    c10::complex<T> gradient,
    c10::complex<T> z
) {
  c10::complex<T> w_z = faddeeva_w(z);

  // First derivative: d/dz w(z) = -2z*w(z) + 2i/sqrt(pi)
  c10::complex<T> two_i_sqrt_pi(T(0), detail::faddeeva_second_deriv_constants<T>::two_over_sqrt_pi);
  c10::complex<T> first_deriv = c10::complex<T>(T(-2), T(0)) * z * w_z + two_i_sqrt_pi;

  // Second derivative: d^2/dz^2 w(z) = (4z^2 - 2)*w(z) - 4iz/sqrt(pi)
  c10::complex<T> z_sq = z * z;
  // 4iz/sqrt(pi) = 4/sqrt(pi) * i * z = 4/sqrt(pi) * (-y + ix) where z = x + iy
  T coeff = detail::faddeeva_second_deriv_constants<T>::four_over_sqrt_pi;
  c10::complex<T> four_i_z_sqrt_pi(-coeff * z.imag(), coeff * z.real());
  c10::complex<T> second_deriv = (c10::complex<T>(T(4), T(0)) * z_sq - c10::complex<T>(T(2), T(0))) * w_z - four_i_z_sqrt_pi;

  // First output: gradient w.r.t. the incoming gradient
  // d(backward)/d(gradient) = conj(first_deriv)
  c10::complex<T> grad_gradient = gradient_gradient * std::conj(first_deriv);

  // Second output: gradient w.r.t. z
  // This is gradient_gradient * gradient * conj(second_deriv)
  c10::complex<T> grad_z = gradient_gradient * gradient * std::conj(second_deriv);

  return {grad_gradient, grad_z};
}

}  // namespace torchscience::kernel::special_functions
