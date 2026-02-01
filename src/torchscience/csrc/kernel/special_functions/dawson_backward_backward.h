#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "dawson.h"
#include "dawson_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order gradient of Dawson's integral
//
// First derivative: D'(z) = 1 - 2z * D(z)
// Second derivative: D''(z) = d/dz[1 - 2z * D(z)]
//                          = -2 * D(z) - 2z * D'(z)
//                          = -2 * D(z) - 2z * (1 - 2z * D(z))
//                          = -2 * D(z) - 2z + 4z^2 * D(z)
//                          = (4z^2 - 2) * D(z) - 2z
//
// For the backward_backward, we need to compute:
//   gg_out = gg_z * d(grad_z)/d(grad_output) = gg_z * D'(z)
//   new_grad_z = gg_z * grad_output * D''(z)
template <typename T>
std::tuple<T, T> dawson_backward_backward(T gg_z, T gradient, T z) {
  T D_z = dawson(z);
  T dD_dz = T(1) - T(2) * z * D_z;

  // gg_out: contribution to gradient of grad_output
  T gg_out = gg_z * dD_dz;

  // d^2D/dz^2 = (4z^2 - 2) * D(z) - 2z
  T d2D_dz2 = (T(4) * z * z - T(2)) * D_z - T(2) * z;

  // new_grad_z: contribution to gradient of z
  T new_grad_z = gg_z * gradient * d2D_dz2;

  return {gg_out, new_grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> dawson_backward_backward(
    c10::complex<T> gg_z,
    c10::complex<T> gradient,
    c10::complex<T> z) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> four(T(4), T(0));

  c10::complex<T> D_z = dawson(z);
  c10::complex<T> dD_dz = one - two * z * D_z;

  // gg_out: For complex, we need to use Wirtinger convention
  c10::complex<T> gg_out = gg_z * std::conj(dD_dz);

  // d^2D/dz^2 = (4z^2 - 2) * D(z) - 2z
  c10::complex<T> d2D_dz2 = (four * z * z - two) * D_z - two * z;

  // new_grad_z: apply Wirtinger convention
  c10::complex<T> new_grad_z = gg_z * gradient * std::conj(d2D_dz2);

  return {gg_out, new_grad_z};
}

}  // namespace torchscience::kernel::special_functions
