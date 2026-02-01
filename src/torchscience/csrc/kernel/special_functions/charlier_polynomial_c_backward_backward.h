#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "charlier_polynomial_c.h"
#include "charlier_polynomial_c_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Charlier polynomial C_n(x; a)
//
// Given:
//   grad_n, grad_x, grad_a = backward(gradient, n, x, a)
//
// We need to compute:
//   d(grad_n)/d*, d(grad_x)/d*, d(grad_a)/d*
//
// where * iterates over n, x, a, gradient
//
// Uses finite differences for second derivatives

template <typename T>
std::tuple<T, T, T, T> charlier_polynomial_c_backward_backward(
    T gg_n, T gg_x, T gg_a,
    T gradient, T n, T x, T a) {

  T eps = T(1e-5);
  T gradient_gradient_output = T(0);
  T new_grad_n = T(0);
  T new_grad_x = T(0);
  T new_grad_a = T(0);

  // Contribution from gg_n (second derivatives involving n)
  if (std::abs(gg_n) > T(1e-15)) {
    // d(grad_n)/d_gradient = dC/dn
    auto [gn, gx, ga] = charlier_polynomial_c_backward(T(1), n, x, a);
    gradient_gradient_output += gg_n * gn;

    // d(grad_n)/dn using finite difference
    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    T d2C_dn2 = (gn_plus - gn_minus) / (T(2) * eps);
    new_grad_n += gg_n * d2C_dn2;

    // d(grad_n)/dx using finite difference
    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    T d2C_dndx = (gn_x_plus - gn_x_minus) / (T(2) * eps);
    new_grad_x += gg_n * d2C_dndx;

    // d(grad_n)/da using finite difference
    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    T d2C_dnda = (gn_a_plus - gn_a_minus) / (T(2) * eps);
    new_grad_a += gg_n * d2C_dnda;
  }

  // Contribution from gg_x (second derivatives involving x)
  if (std::abs(gg_x) > T(1e-15)) {
    // d(grad_x)/d_gradient = dC/dx
    auto [gn, gx, ga] = charlier_polynomial_c_backward(T(1), n, x, a);
    gradient_gradient_output += gg_x * gx;

    // d(grad_x)/dn using finite difference
    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    T d2C_dxdn = (gx_plus - gx_minus) / (T(2) * eps);
    new_grad_n += gg_x * d2C_dxdn;

    // d(grad_x)/dx using finite difference
    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    T d2C_dx2 = (gx_x_plus - gx_x_minus) / (T(2) * eps);
    new_grad_x += gg_x * d2C_dx2;

    // d(grad_x)/da using finite difference
    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    T d2C_dxda = (gx_a_plus - gx_a_minus) / (T(2) * eps);
    new_grad_a += gg_x * d2C_dxda;
  }

  // Contribution from gg_a (second derivatives involving a)
  if (std::abs(gg_a) > T(1e-15)) {
    // d(grad_a)/d_gradient = dC/da
    auto [gn, gx, ga] = charlier_polynomial_c_backward(T(1), n, x, a);
    gradient_gradient_output += gg_a * ga;

    // d(grad_a)/dn using finite difference
    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    T d2C_dadn = (ga_plus - ga_minus) / (T(2) * eps);
    new_grad_n += gg_a * d2C_dadn;

    // d(grad_a)/dx using finite difference
    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    T d2C_dadx = (gx_x_plus - gx_x_minus) / (T(2) * eps);
    new_grad_x += gg_a * d2C_dadx;

    // d(grad_a)/da using finite difference
    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    T d2C_da2 = (ga_a_plus - ga_a_minus) / (T(2) * eps);
    new_grad_a += gg_a * d2C_da2;
  }

  return {gradient_gradient_output, new_grad_n, new_grad_x, new_grad_a};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
charlier_polynomial_c_backward_backward(
    c10::complex<T> gg_n, c10::complex<T> gg_x, c10::complex<T> gg_a,
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> x, c10::complex<T> a) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_n = zero;
  c10::complex<T> new_grad_x = zero;
  c10::complex<T> new_grad_a = zero;

  // Contribution from gg_n
  if (std::abs(gg_n) > eps_val) {
    auto [gn, gx, ga] = charlier_polynomial_c_backward(one, n, x, a);
    gradient_gradient_output += gg_n * std::conj(gn);

    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    c10::complex<T> d2C_dn2 = (gn_plus - gn_minus) / (two * eps);
    new_grad_n += gg_n * std::conj(d2C_dn2);

    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    c10::complex<T> d2C_dndx = (gn_x_plus - gn_x_minus) / (two * eps);
    new_grad_x += gg_n * std::conj(d2C_dndx);

    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    c10::complex<T> d2C_dnda = (gn_a_plus - gn_a_minus) / (two * eps);
    new_grad_a += gg_n * std::conj(d2C_dnda);
  }

  // Contribution from gg_x
  if (std::abs(gg_x) > eps_val) {
    auto [gn, gx, ga] = charlier_polynomial_c_backward(one, n, x, a);
    gradient_gradient_output += gg_x * std::conj(gx);

    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    c10::complex<T> d2C_dxdn = (gx_plus - gx_minus) / (two * eps);
    new_grad_n += gg_x * std::conj(d2C_dxdn);

    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    c10::complex<T> d2C_dx2 = (gx_x_plus - gx_x_minus) / (two * eps);
    new_grad_x += gg_x * std::conj(d2C_dx2);

    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    c10::complex<T> d2C_dxda = (gx_a_plus - gx_a_minus) / (two * eps);
    new_grad_a += gg_x * std::conj(d2C_dxda);
  }

  // Contribution from gg_a
  if (std::abs(gg_a) > eps_val) {
    auto [gn, gx, ga] = charlier_polynomial_c_backward(one, n, x, a);
    gradient_gradient_output += gg_a * std::conj(ga);

    auto [gn_plus, gx_plus, ga_plus] = charlier_polynomial_c_backward(gradient, n + eps, x, a);
    auto [gn_minus, gx_minus, ga_minus] = charlier_polynomial_c_backward(gradient, n - eps, x, a);
    c10::complex<T> d2C_dadn = (ga_plus - ga_minus) / (two * eps);
    new_grad_n += gg_a * std::conj(d2C_dadn);

    auto [gn_x_plus, gx_x_plus, ga_x_plus] = charlier_polynomial_c_backward(gradient, n, x + eps, a);
    auto [gn_x_minus, gx_x_minus, ga_x_minus] = charlier_polynomial_c_backward(gradient, n, x - eps, a);
    c10::complex<T> d2C_dadx = (ga_x_plus - ga_x_minus) / (two * eps);
    new_grad_x += gg_a * std::conj(d2C_dadx);

    auto [gn_a_plus, gx_a_plus, ga_a_plus] = charlier_polynomial_c_backward(gradient, n, x, a + eps);
    auto [gn_a_minus, gx_a_minus, ga_a_minus] = charlier_polynomial_c_backward(gradient, n, x, a - eps);
    c10::complex<T> d2C_da2 = (ga_a_plus - ga_a_minus) / (two * eps);
    new_grad_a += gg_a * std::conj(d2C_da2);
  }

  return {gradient_gradient_output, new_grad_n, new_grad_x, new_grad_a};
}

} // namespace torchscience::kernel::special_functions
