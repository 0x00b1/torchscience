#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cos_pi.h"
#include "tan_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> tan_pi_backward_backward(T gradient_gradient, T gradient, T x) {
  T pi = static_cast<T>(M_PI);

  T cos_pi_x = cos_pi(x);

  T sec2_pi_x = T(1) / (cos_pi_x * cos_pi_x);

  return std::make_tuple(gradient_gradient * pi * sec2_pi_x, gradient_gradient * gradient * T(2) * pi * pi * tan_pi(x) * sec2_pi_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> tan_pi_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> cos_pi_z = cos_pi(z);

  c10::complex<T> sec2_pi_z = c10::complex<T>(T(1), T(0)) / (cos_pi_z * cos_pi_z);

  c10::complex<T> gg_output = gradient_gradient * pi * sec2_pi_z;
  c10::complex<T> g_z = gradient_gradient * gradient * T(2) * pi * pi * tan_pi(z) * sec2_pi_z;

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
