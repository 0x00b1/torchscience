#pragma once

#include <tuple>

#include "digamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> log_beta_backward_backward(
  T gradient_gradient_a,
  T gradient_gradient_b,
  T gradient,
  T a,
  T b
) {
  T psi_ab = digamma(a + b);
  T psi1_ab = trigamma(a + b);

  T diff_a = digamma(a) - psi_ab;
  T diff_b = digamma(b) - psi_ab;

  // Gradient w.r.t. grad_output
  T gg_output = gradient_gradient_a * diff_a + gradient_gradient_b * diff_b;

  // Second derivatives:
  // d²/da² log_beta = trigamma(a) - trigamma(a+b)
  // d²/db² log_beta = trigamma(b) - trigamma(a+b)
  // d²/dadb log_beta = -trigamma(a+b)
  T d2_aa = trigamma(a) - psi1_ab;
  T d2_bb = trigamma(b) - psi1_ab;
  T d2_ab = -psi1_ab;

  return {
    gg_output,
    gradient_gradient_a * (gradient * d2_aa) + gradient_gradient_b * (gradient * d2_ab),
    gradient_gradient_a * (gradient * d2_ab) + gradient_gradient_b * (gradient * d2_bb)
  };
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
log_beta_backward_backward(
  c10::complex<T> gradient_gradient_a,
  c10::complex<T> gradient_gradient_b,
  c10::complex<T> gradient,
  c10::complex<T> a,
  c10::complex<T> b
) {
  c10::complex<T> psi_ab = digamma(a + b);
  c10::complex<T> psi1_ab = trigamma(a + b);

  c10::complex<T> diff_a = digamma(a) - psi_ab;
  c10::complex<T> diff_b = digamma(b) - psi_ab;

  c10::complex<T> gg_output = gradient_gradient_a * diff_a + gradient_gradient_b * diff_b;

  c10::complex<T> d2_aa = trigamma(a) - psi1_ab;
  c10::complex<T> d2_bb = trigamma(b) - psi1_ab;
  c10::complex<T> d2_ab = -psi1_ab;

  return {
    gg_output,
    gradient_gradient_a * (gradient * d2_aa) + gradient_gradient_b * (gradient * d2_ab),
    gradient_gradient_a * (gradient * d2_ab) + gradient_gradient_b * (gradient * d2_bb)
  };
}

} // namespace torchscience::kernel::special_functions
