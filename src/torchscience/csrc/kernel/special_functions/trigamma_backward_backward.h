#pragma once

#include <tuple>

#include "tetragamma.h"
#include "pentagamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> trigamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * tetragamma(z),
    gradient_gradient * gradient * pentagamma(z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> trigamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * tetragamma(z),
    gradient_gradient * gradient * pentagamma(z)
  };
}

} // namespace torchscience::kernel::special_functions
