#pragma once

#include <tuple>

#include "beta.h"
#include "digamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> beta_backward_backward(
  T gradient_gradient_a,
  T gradient_gradient_b,
  T gradient,
  T a,
  T b
) {
  T beta_ab = beta(a, b);
  T psi_ab = digamma(a + b);
  T psi1_ab = trigamma(a + b);

  T diff_a = digamma(a) - psi_ab;
  T diff_b = digamma(b) - psi_ab;

  T gg_output = gradient_gradient_a * beta_ab * diff_a + gradient_gradient_b * beta_ab * diff_b;

  return {
    gg_output,
    gradient_gradient_a * (gradient * beta_ab * (diff_a * diff_a + trigamma(a) - psi1_ab)) + gradient_gradient_b * (gradient * beta_ab * (diff_a * diff_b - psi1_ab)),
    gradient_gradient_a * (gradient * beta_ab * (diff_a * diff_b - psi1_ab)) + gradient_gradient_b * (gradient * beta_ab * (diff_b * diff_b + trigamma(b) - psi1_ab))
  };
}

template <typename T>
std::tuple<
  c10::complex<T>,
  c10::complex<T>,
  c10::complex<T>
> beta_backward_backward(
  c10::complex<T> gradient_gradient_a,
  c10::complex<T> gradient_gradient_b,
  c10::complex<T> gradient,
  c10::complex<T> a,
  c10::complex<T> b
) {
  c10::complex<T> beta_ab = beta(a, b);
  c10::complex<T> psi_ab = digamma(a + b);
  c10::complex<T> psi1_ab = trigamma(a + b);

  c10::complex<T> diff_a = digamma(a) - psi_ab;
  c10::complex<T> diff_b = digamma(b) - psi_ab;

  c10::complex<T> d2_aa = gradient * beta_ab * (diff_a * diff_a + trigamma(a) - psi1_ab);
  c10::complex<T> d2_ab = gradient * beta_ab * (diff_a * diff_b - psi1_ab);
  c10::complex<T> d2_bb = gradient * beta_ab * (diff_b * diff_b + trigamma(b) - psi1_ab);

  return {
    gradient_gradient_a * beta_ab * diff_a + gradient_gradient_b * beta_ab * diff_b,
    gradient_gradient_a * d2_aa + gradient_gradient_b * d2_ab,
    gradient_gradient_a * d2_ab + gradient_gradient_b * d2_bb
  };
}

} // namespace torchscience::kernel::special_functions
