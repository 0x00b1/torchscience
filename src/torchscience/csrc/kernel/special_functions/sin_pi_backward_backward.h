#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "cos_pi.h"
#include "sin_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> sin_pi_backward_backward(T gradient_gradient, T gradient, T x) {
  T x_mod = std::fmod(x, T(2));

  if (x_mod < T(0)) {
    x_mod += T(2);
  }

  T pi = static_cast<T>(M_PI);
  T pi_x = pi * x_mod;

  T g_x = -gradient_gradient * gradient * pi * pi * std::sin(pi_x);

  return std::make_tuple(gradient_gradient * pi * std::cos(pi_x), g_x);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> sin_pi_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);

  c10::complex<T> gg_output = gradient_gradient * pi * cos_pi(z);
  c10::complex<T> g_z = -gradient_gradient * gradient * pi * pi * sin_pi(z);

  return std::make_tuple(gg_output, g_z);
}

} // namespace torchscience::kernel::special_functions
