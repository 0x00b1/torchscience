#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "tanh_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> tanh_pi_backward_backward(T gradient_gradient, T gradient, T x) {
  T pi = static_cast<T>(M_PI);

  T tanh_pi_x = tanh_pi(x);
  T sech2_pi_x = T(1) - tanh_pi_x * tanh_pi_x;

  return std::make_tuple(
    gradient_gradient * pi * sech2_pi_x,
    -gradient_gradient * gradient * T(2) * pi * pi * tanh_pi_x * sech2_pi_x
  );
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> tanh_pi_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  T pi = static_cast<T>(M_PI);
  c10::complex<T> one(T(1), T(0));

  c10::complex<T> tanh_pi_z = tanh_pi(z);
  c10::complex<T> sech2_pi_z = one - tanh_pi_z * tanh_pi_z;

  return std::make_tuple(
    gradient_gradient * pi * sech2_pi_z,
    -gradient_gradient * gradient * T(2) * pi * pi * tanh_pi_z * sech2_pi_z
  );
}

} // namespace torchscience::kernel::special_functions
