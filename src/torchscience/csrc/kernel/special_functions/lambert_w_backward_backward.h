#pragma once

#include <tuple>

#include "lambert_w.h"

namespace torchscience::kernel::special_functions {

// Second-order backward for Lambert W function
//
// Forward: y = W_k(z)
// Backward: grad_k = 0, grad_z = grad_output * dW/dz
//           where dW/dz = W(z) / (z * (1 + W(z)))
//
// backward_backward computes gradients of backward outputs w.r.t. inputs:
// - d(grad_k)/d(anything) = 0 since grad_k = 0
// - d(grad_z)/d(grad_output) = dW/dz = W(z) / (z * (1 + W(z)))
// - d(grad_z)/d(k) = 0 (k is discrete)
// - d(grad_z)/d(z) = grad_output * d^2W/dz^2
//
// Second derivative derivation:
// From W' = W / (z * (1 + W)), using quotient rule:
// W'' = (W' * z * (1 + W) - W * (1 + W + z*W')) / (z * (1 + W))^2
//     = (W - W * (1 + 3W + W^2) / (1 + W)) / (z^2 * (1 + W)^2)
//     = -W^2 * (2 + W) / (z^2 * (1 + W)^3)
//
// where d^2W/dz^2 = -W^2 * (2 + W) / (z^2 * (1 + W)^3)
//
// Returns: (grad_grad_output, grad_k, grad_z)
// grad_grad_output = gg_z * dW/dz (contribution to grad_output gradient)
// grad_k = 0 (k is discrete)
// grad_z = gg_z * grad_output * d^2W/dz^2 (contribution to z gradient)
template <typename T>
std::tuple<T, T, T> lambert_w_backward_backward(
    T gg_k,       // upstream gradient for grad_k output (unused since grad_k = 0)
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T k,
    T z) {
  // Since grad_k = 0, gg_k does not contribute to any gradient
  (void)gg_k;

  const T eps = detail::lambert_w_eps<T>();

  // Handle z = 0 case
  // At z = 0: W(0) = 0, dW/dz|_{z=0} = 1, d^2W/dz^2|_{z=0} = -2
  if (std::abs(z) < eps) {
    // grad_grad_output = gg_z * dW/dz = gg_z * 1 = gg_z
    // grad_k = 0
    // grad_z = gg_z * grad_output * d^2W/dz^2 = gg_z * grad_output * (-2)
    return {gg_z, T(0), T(-2) * grad_output * gg_z};
  }

  // Compute W(k, z)
  T w = lambert_w(k, z);

  // Handle NaN case
  if (std::isnan(w)) {
    T nan = std::numeric_limits<T>::quiet_NaN();
    return {nan, T(0), nan};
  }

  T wp1 = w + T(1);

  // Handle w = -1 case (branch point): derivatives are singular
  if (std::abs(wp1) < eps) {
    T inf = std::numeric_limits<T>::infinity();
    return {inf, T(0), inf};
  }

  // First derivative: dW/dz = W(z) / (z * (1 + W(z)))
  T dW_dz = w / (z * wp1);

  // Second derivative: d^2W/dz^2 = -W^2 * (2 + W) / (z^2 * (1 + W)^3)
  T wp1_cubed = wp1 * wp1 * wp1;
  T d2W_dz2 = -w * w * (T(2) + w) / (z * z * wp1_cubed);

  // grad_grad_output = gg_z * dW/dz (from grad_z = grad_output * dW/dz)
  T grad_grad_output = gg_z * dW_dz;

  // grad_k = 0 (k is discrete)
  T grad_k = T(0);

  // grad_z = gg_z * grad_output * d^2W/dz^2
  T grad_z = gg_z * grad_output * d2W_dz2;

  return {grad_grad_output, grad_k, grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> lambert_w_backward_backward(
    c10::complex<T> gg_k,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> k,
    c10::complex<T> z) {
  // Since grad_k = 0, gg_k does not contribute
  (void)gg_k;

  const T eps = detail::lambert_w_eps<T>();
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> zero(T(0), T(0));

  // Handle z = 0 case
  if (std::abs(z) < eps) {
    return {gg_z, zero, -two * grad_output * gg_z};
  }

  // Compute W(k, z)
  c10::complex<T> w = lambert_w(k, z);

  c10::complex<T> wp1 = w + one;

  // Handle w = -1 case (branch point)
  if (std::abs(wp1) < eps) {
    c10::complex<T> inf(std::numeric_limits<T>::infinity(), T(0));
    return {inf, zero, inf};
  }

  // First derivative: dW/dz = W(z) / (z * (1 + W(z)))
  c10::complex<T> dW_dz = w / (z * wp1);

  // Second derivative: d^2W/dz^2 = -W^2 * (2 + W) / (z^2 * (1 + W)^3)
  c10::complex<T> wp1_cubed = wp1 * wp1 * wp1;
  c10::complex<T> d2W_dz2 = -w * w * (two + w) / (z * z * wp1_cubed);

  // grad_grad_output = gg_z * dW/dz
  c10::complex<T> grad_grad_output = gg_z * dW_dz;

  // grad_k = 0
  c10::complex<T> grad_k = zero;

  // grad_z = gg_z * grad_output * d^2W/dz^2
  c10::complex<T> grad_z = gg_z * grad_output * d2W_dz2;

  return {grad_grad_output, grad_k, grad_z};
}

} // namespace torchscience::kernel::special_functions
