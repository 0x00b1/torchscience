#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "digamma.h"
#include "trigamma.h"
#include "pochhammer.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> pochhammer_backward_backward(
    T gg_z, T gg_m, T grad_output, T z, T m) {
  T poch = pochhammer(z, m);
  T psi_zm = digamma(z + m);
  T psi_z = digamma(z);
  T tri_zm = trigamma(z + m);
  T tri_z = trigamma(z);

  T diff_psi = psi_zm - psi_z;

  T grad_grad_output = gg_z * poch * diff_psi + gg_m * poch * psi_zm;

  T d2_dz2 = poch * (diff_psi * diff_psi + tri_zm - tri_z);
  T d2_dm2 = poch * (psi_zm * psi_zm + tri_zm);
  T d2_dzdm = poch * (diff_psi * psi_zm + tri_zm);

  T grad_z = grad_output * (gg_z * d2_dz2 + gg_m * d2_dzdm);
  T grad_m = grad_output * (gg_z * d2_dzdm + gg_m * d2_dm2);

  return std::make_tuple(grad_grad_output, grad_z, grad_m);
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> pochhammer_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> gg_m, c10::complex<T> grad_output,
    c10::complex<T> z, c10::complex<T> m) {
  c10::complex<T> poch = pochhammer(z, m);
  c10::complex<T> psi_zm = digamma(z + m);
  c10::complex<T> psi_z = digamma(z);
  c10::complex<T> tri_zm = trigamma(z + m);
  c10::complex<T> tri_z = trigamma(z);

  c10::complex<T> diff_psi = psi_zm - psi_z;

  c10::complex<T> grad_grad_output = gg_z * poch * diff_psi + gg_m * poch * psi_zm;

  c10::complex<T> d2_dz2 = poch * (diff_psi * diff_psi + tri_zm - tri_z);
  c10::complex<T> d2_dm2 = poch * (psi_zm * psi_zm + tri_zm);
  c10::complex<T> d2_dzdm = poch * (diff_psi * psi_zm + tri_zm);

  c10::complex<T> grad_z = grad_output * (gg_z * d2_dz2 + gg_m * d2_dzdm);
  c10::complex<T> grad_m = grad_output * (gg_z * d2_dzdm + gg_m * d2_dm2);

  return std::make_tuple(grad_grad_output, grad_z, grad_m);
}

} // namespace torchscience::kernel::special_functions
