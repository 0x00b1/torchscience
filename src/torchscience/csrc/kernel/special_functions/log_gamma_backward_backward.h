#pragma once

#include <tuple>

#include "digamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> log_gamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  return {
    gradient_gradient * digamma(z),
    gradient_gradient * gradient * trigamma(z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> log_gamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  return {
    gradient_gradient * digamma(z),
    gradient_gradient * gradient * trigamma(z)
  };
}

} // namespace torchscience::kernel::special_functions
