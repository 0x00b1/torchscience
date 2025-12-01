#pragma once

#include <tuple>

#include "digamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> log_beta_backward(T gradient, T a, T b) {
  T psi_ab = digamma(a + b);
  return {
    gradient * (digamma(a) - psi_ab),
    gradient * (digamma(b) - psi_ab)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> log_beta_backward(
  c10::complex<T> gradient,
  c10::complex<T> a,
  c10::complex<T> b
) {
  c10::complex<T> psi_ab = digamma(a + b);
  return {
    gradient * (digamma(a) - psi_ab),
    gradient * (digamma(b) - psi_ab)
  };
}

} // namespace torchscience::kernel::special_functions
