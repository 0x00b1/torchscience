#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "digamma.h"
#include "gamma.h"
#include "trigamma.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
template <typename T>
std::tuple<T, T> gamma_backward_backward(
  T gradient_gradient,
  T gradient,
  T z
) {
  T gamma_z = gamma(z);

  T psi = digamma(z);

  return {
    gradient_gradient * gamma_z * psi,
    gradient_gradient * gradient * gamma_z * (psi * psi + trigamma(z))
  };
}

// Complex backward_backward: apply conjugation for Wirtinger derivatives
// PyTorch convention: grad * conj(derivative) for holomorphic functions
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> gamma_backward_backward(
  c10::complex<T> gradient_gradient,
  c10::complex<T> gradient,
  c10::complex<T> z
) {
  c10::complex<T> gamma_z = gamma(z);
  c10::complex<T> psi = digamma(z);
  c10::complex<T> psi1 = trigamma(z);

  // First output: gradient w.r.t. the incoming gradient (for chain rule)
  // d(backward)/d(gradient) = conj(Γ(z) * ψ(z))
  c10::complex<T> grad_gradient = gradient_gradient * std::conj(gamma_z * psi);

  // Second output: gradient w.r.t. z (second derivative term)
  // d²Γ/dz² = Γ(z) * (ψ(z)² + ψ'(z))
  c10::complex<T> grad_z = gradient_gradient * gradient * std::conj(gamma_z * (psi * psi + psi1));

  return {grad_gradient, grad_z};
}

} // namespace torchscience::kernel::special_functions
