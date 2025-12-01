#pragma once

#include <tuple>

#include "trigamma.h"
#include "tetragamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> digamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * trigamma(z),
    gradient_gradient * gradient * tetragamma(z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> digamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * trigamma(z),
    gradient_gradient * gradient * tetragamma(z)
  };
}

} // namespace torchscience::kernel::special_functions
