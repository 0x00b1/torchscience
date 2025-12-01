#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cosh_pi.h"
#include "sinh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> sinh_pi_backward_backward(T gradient_gradient, T gradient, T x) {
  T pi = static_cast<T>(M_PI);
  T pi_x = pi * x;

  T sinh_pi_x = std::sinh(pi_x);
  T cosh_pi_x = std::cosh(pi_x);

  T gg_output = gradient_gradient * pi * cosh_pi_x;
  T g_x = gradient_gradient * gradient * pi * pi * sinh_pi_x;

  return std::make_tuple(gg_output, g_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> sinh_pi_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> gg_output = gradient_gradient * pi * cosh_pi(z);
  c10::complex<T> g_z = gradient_gradient * gradient * pi * pi * sinh_pi(z);

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
