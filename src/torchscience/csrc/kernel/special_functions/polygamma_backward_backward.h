#pragma once

#include <tuple>

#include "polygamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> polygamma_backward_backward(
  T gradient_gradient_n,
  T gradient_gradient_z,
  T gradient,
  T n,
  T z
) {
  return {
    gradient_gradient_z * polygamma(n + T(1), z),
    T(0),
    gradient_gradient_z * gradient * polygamma(n + T(2), z)};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> polygamma_backward_backward(
  c10::complex<T> gradient_gradient_n,
  c10::complex<T> gradient_gradient_z,
  c10::complex<T> gradient,
  c10::complex<T> n,
  c10::complex<T> z
) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  return {
    gradient_gradient_z * polygamma(n + one, z),
    c10::complex<T>(T(0), T(0)),
    gradient_gradient_z * gradient * polygamma(n + two, z)
  };
}

} // namespace torchscience::kernel::special_functions
