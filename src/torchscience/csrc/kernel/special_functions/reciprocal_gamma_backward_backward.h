#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "digamma.h"
#include "trigamma.h"
#include "reciprocal_gamma.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> reciprocal_gamma_backward_backward(T gg_z, T grad_output, T z) {
  T rg = reciprocal_gamma(z);
  T psi = digamma(z);
  T psi_sq = psi * psi;
  T tri = trigamma(z);

  T grad_grad_output = gg_z * (-psi * rg);
  T grad_z = gg_z * grad_output * rg * (psi_sq - tri);

  return std::make_tuple(grad_grad_output, grad_z);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> reciprocal_gamma_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
  c10::complex<T> rg = reciprocal_gamma(z);
  c10::complex<T> psi = digamma(z);
  c10::complex<T> psi_sq = psi * psi;
  c10::complex<T> tri = trigamma(z);

  c10::complex<T> grad_grad_output = gg_z * (-psi * rg);
  c10::complex<T> grad_z = gg_z * grad_output * rg * (psi_sq - tri);

  return std::make_tuple(grad_grad_output, grad_z);
}

} // namespace torchscience::kernel::special_functions
