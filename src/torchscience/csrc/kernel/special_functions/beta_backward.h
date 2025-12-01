#pragma once

#include <tuple>

#include "beta.h"
#include "digamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<
  T,
  T
> beta_backward(
  T gradient,
  T a,
  T b
) {
  return {
    gradient * beta(a, b) * (digamma(a) - digamma(a + b)),
    gradient * beta(a, b) * (digamma(b) - digamma(a + b))
  };
}

template <typename T>
std::tuple<
  c10::complex<T>,
  c10::complex<T>
> beta_backward(
  c10::complex<T> gradient,
  c10::complex<T> a,
  c10::complex<T> b
) {
  return {
    gradient * beta(a, b) * (digamma(a) - digamma(a + b)),
    gradient * beta(a, b) * (digamma(b) - digamma(a + b))
  };
}

} // namespace torchscience::kernel::special_functions
