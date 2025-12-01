#pragma once

#include <tuple>

#include "polygamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> polygamma_backward(
  T gradient,
  T n,
  T z
) {
  return {
    T(0),
    gradient * polygamma(n + T(1), z)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> polygamma_backward(
  c10::complex<T> gradient,
  c10::complex<T> n,
  c10::complex<T> z
) {
  return {
    c10::complex<T>(T(0), T(0)),
    gradient * polygamma(n + c10::complex<T>(T(1), T(0)), z)
  };
}

} // namespace torchscience::kernel::special_functions
