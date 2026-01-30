#pragma once

#include <tuple>

#include "lambert_w.h"

namespace torchscience::kernel::special_functions {

// Derivative of Lambert W function with respect to z:
// d/dz W(z) = W(z) / (z * (1 + W(z))) for z != 0
// d/dz W(z) = 1 for z = 0
//
// For the backward pass with grad_output g:
// grad_k = 0 (k is discrete, no gradient)
// grad_z = g * W(z) / (z * (1 + W(z)))
template <typename T>
std::tuple<T, T> lambert_w_backward(T gradient, T k, T z) {
  // k is a branch index parameter - no gradient with respect to it
  T grad_k = T(0);

  // Handle z = 0 case: derivative is 1
  const T eps = detail::lambert_w_eps<T>();
  if (std::abs(z) < eps) {
    return {grad_k, gradient};
  }

  // Compute W(k, z)
  T w = lambert_w(k, z);

  // Handle NaN case
  if (std::isnan(w)) {
    return {grad_k, std::numeric_limits<T>::quiet_NaN()};
  }

  // Handle w = -1 case (branch point): derivative is infinite
  T wp1 = w + T(1);
  if (std::abs(wp1) < eps) {
    return {grad_k, std::numeric_limits<T>::infinity()};
  }

  // d/dz W(z) = W(z) / (z * (1 + W(z)))
  T dw_dz = w / (z * wp1);

  return {grad_k, gradient * dw_dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> lambert_w_backward(
    c10::complex<T> gradient,
    c10::complex<T> k,
    c10::complex<T> z) {
  // k is a branch index parameter - no gradient with respect to it
  c10::complex<T> grad_k(T(0), T(0));
  c10::complex<T> one(T(1), T(0));

  // Handle z = 0 case: derivative is 1
  const T eps = detail::lambert_w_eps<T>();
  if (std::abs(z) < eps) {
    return {grad_k, gradient};
  }

  // Compute W(k, z)
  c10::complex<T> w = lambert_w(k, z);

  // Handle w = -1 case (branch point)
  c10::complex<T> wp1 = w + one;
  if (std::abs(wp1) < eps) {
    // Return infinity for singular derivative
    return {grad_k, c10::complex<T>(std::numeric_limits<T>::infinity(), T(0))};
  }

  // d/dz W(z) = W(z) / (z * (1 + W(z)))
  c10::complex<T> dw_dz = w / (z * wp1);

  return {grad_k, gradient * dw_dz};
}

} // namespace torchscience::kernel::special_functions
