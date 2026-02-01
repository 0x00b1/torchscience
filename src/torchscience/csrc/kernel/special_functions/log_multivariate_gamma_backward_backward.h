#pragma once

#include <tuple>

#include "digamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> log_multivariate_gamma_backward_backward(
    T gg_a, T grad_output, T a, int64_t d) {
  T sum_psi = T(0);
  T sum_tri = T(0);
  for (int64_t j = 1; j <= d; ++j) {
    T arg = a + (T(1) - static_cast<T>(j)) / T(2);
    sum_psi += digamma(arg);
    sum_tri += trigamma(arg);
  }

  T grad_grad_output = gg_a * sum_psi;
  T grad_a = gg_a * grad_output * sum_tri;

  return std::make_tuple(grad_grad_output, grad_a);
}

} // namespace torchscience::kernel::special_functions
