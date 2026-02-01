#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "laguerre_polynomial_l.h"
#include "laguerre_polynomial_l_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for generalized Laguerre polynomial L_n^alpha(z)
//
// Given:
//   grad_n, grad_alpha, grad_z = backward(gradient, n, alpha, z)
//
// We need to compute:
//   d(grad_n)/d*, d(grad_alpha)/d*, d(grad_z)/d*
//
// where * iterates over n, alpha, z, gradient
//
// Uses finite differences for second derivatives

template <typename T>
std::tuple<T, T, T, T> laguerre_polynomial_l_backward_backward(
    T gg_n, T gg_alpha, T gg_z,
    T gradient, T n, T alpha, T z) {

  T eps = T(1e-5);
  T gradient_gradient_output = T(0);
  T new_grad_n = T(0);
  T new_grad_alpha = T(0);
  T new_grad_z = T(0);

  // Contribution from gg_n (second derivatives involving n)
  if (std::abs(gg_n) > T(1e-15)) {
    // d(grad_n)/d_gradient = d(gradient * dL/dn)/d_gradient = dL/dn
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(T(1), n, alpha, z);
    gradient_gradient_output += gg_n * gn;

    // d(grad_n)/dn using finite difference
    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    T d2L_dn2 = (gn_plus - gn_minus) / (T(2) * eps);
    new_grad_n += gg_n * d2L_dn2;

    // d(grad_n)/dalpha using finite difference
    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    T d2L_dndalpha = (gn_alpha_plus - gn_alpha_minus) / (T(2) * eps);
    new_grad_alpha += gg_n * d2L_dndalpha;

    // d(grad_n)/dz using finite difference
    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    T d2L_dndz = (gn_z_plus - gn_z_minus) / (T(2) * eps);
    new_grad_z += gg_n * d2L_dndz;
  }

  // Contribution from gg_alpha (second derivatives involving alpha)
  if (std::abs(gg_alpha) > T(1e-15)) {
    // d(grad_alpha)/d_gradient = dL/dalpha
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(T(1), n, alpha, z);
    gradient_gradient_output += gg_alpha * ga;

    // d(grad_alpha)/dn using finite difference
    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    T d2L_dalphadn = (ga_plus - ga_minus) / (T(2) * eps);
    new_grad_n += gg_alpha * d2L_dalphadn;

    // d(grad_alpha)/dalpha using finite difference
    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    T d2L_dalpha2 = (ga_alpha_plus - ga_alpha_minus) / (T(2) * eps);
    new_grad_alpha += gg_alpha * d2L_dalpha2;

    // d(grad_alpha)/dz using finite difference
    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    T d2L_dalphadz = (ga_z_plus - ga_z_minus) / (T(2) * eps);
    new_grad_z += gg_alpha * d2L_dalphadz;
  }

  // Contribution from gg_z (second derivatives involving z)
  if (std::abs(gg_z) > T(1e-15)) {
    // d(grad_z)/d_gradient = dL/dz
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(T(1), n, alpha, z);
    gradient_gradient_output += gg_z * gz;

    // d(grad_z)/dn using finite difference
    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    T d2L_dzdn = (gz_plus - gz_minus) / (T(2) * eps);
    new_grad_n += gg_z * d2L_dzdn;

    // d(grad_z)/dalpha using finite difference
    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    T d2L_dzdalpha = (ga_alpha_plus - ga_alpha_minus) / (T(2) * eps);
    new_grad_alpha += gg_z * d2L_dzdalpha;

    // d(grad_z)/dz using finite difference
    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    T d2L_dz2 = (gz_z_plus - gz_z_minus) / (T(2) * eps);
    new_grad_z += gg_z * d2L_dz2;
  }

  return {gradient_gradient_output, new_grad_n, new_grad_alpha, new_grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
laguerre_polynomial_l_backward_backward(
    c10::complex<T> gg_n, c10::complex<T> gg_alpha, c10::complex<T> gg_z,
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> alpha, c10::complex<T> z) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_n = zero;
  c10::complex<T> new_grad_alpha = zero;
  c10::complex<T> new_grad_z = zero;

  // Contribution from gg_n
  if (std::abs(gg_n) > eps_val) {
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(one, n, alpha, z);
    gradient_gradient_output += gg_n * std::conj(gn);

    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    c10::complex<T> d2L_dn2 = (gn_plus - gn_minus) / (two * eps);
    new_grad_n += gg_n * std::conj(d2L_dn2);

    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    c10::complex<T> d2L_dndalpha = (gn_alpha_plus - gn_alpha_minus) / (two * eps);
    new_grad_alpha += gg_n * std::conj(d2L_dndalpha);

    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    c10::complex<T> d2L_dndz = (gn_z_plus - gn_z_minus) / (two * eps);
    new_grad_z += gg_n * std::conj(d2L_dndz);
  }

  // Contribution from gg_alpha
  if (std::abs(gg_alpha) > eps_val) {
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(one, n, alpha, z);
    gradient_gradient_output += gg_alpha * std::conj(ga);

    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    c10::complex<T> d2L_dalphadn = (ga_plus - ga_minus) / (two * eps);
    new_grad_n += gg_alpha * std::conj(d2L_dalphadn);

    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    c10::complex<T> d2L_dalpha2 = (ga_alpha_plus - ga_alpha_minus) / (two * eps);
    new_grad_alpha += gg_alpha * std::conj(d2L_dalpha2);

    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    c10::complex<T> d2L_dalphadz = (ga_z_plus - ga_z_minus) / (two * eps);
    new_grad_z += gg_alpha * std::conj(d2L_dalphadz);
  }

  // Contribution from gg_z
  if (std::abs(gg_z) > eps_val) {
    auto [gn, ga, gz] = laguerre_polynomial_l_backward(one, n, alpha, z);
    gradient_gradient_output += gg_z * std::conj(gz);

    auto [gn_plus, ga_plus, gz_plus] = laguerre_polynomial_l_backward(gradient, n + eps, alpha, z);
    auto [gn_minus, ga_minus, gz_minus] = laguerre_polynomial_l_backward(gradient, n - eps, alpha, z);
    c10::complex<T> d2L_dzdn = (gz_plus - gz_minus) / (two * eps);
    new_grad_n += gg_z * std::conj(d2L_dzdn);

    auto [gn_alpha_plus, ga_alpha_plus, gz_alpha_plus] = laguerre_polynomial_l_backward(gradient, n, alpha + eps, z);
    auto [gn_alpha_minus, ga_alpha_minus, gz_alpha_minus] = laguerre_polynomial_l_backward(gradient, n, alpha - eps, z);
    c10::complex<T> d2L_dzdalpha = (gz_alpha_plus - gz_alpha_minus) / (two * eps);
    new_grad_alpha += gg_z * std::conj(d2L_dzdalpha);

    auto [gn_z_plus, ga_z_plus, gz_z_plus] = laguerre_polynomial_l_backward(gradient, n, alpha, z + eps);
    auto [gn_z_minus, ga_z_minus, gz_z_minus] = laguerre_polynomial_l_backward(gradient, n, alpha, z - eps);
    c10::complex<T> d2L_dz2 = (gz_z_plus - gz_z_minus) / (two * eps);
    new_grad_z += gg_z * std::conj(d2L_dz2);
  }

  return {gradient_gradient_output, new_grad_n, new_grad_alpha, new_grad_z};
}

} // namespace torchscience::kernel::special_functions
