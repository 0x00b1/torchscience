#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "zernike_polynomial_r.h"
#include "zernike_polynomial_r_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for radial Zernike polynomial R_n^m(rho)
//
// Given:
//   grad_n, grad_m, grad_rho = backward(gradient, n, m, rho)
//
// We need to compute:
//   d(grad_n)/d*, d(grad_m)/d*, d(grad_rho)/d*
//
// where * iterates over n, m, rho, gradient
//
// Uses finite differences for second derivatives

template <typename T>
std::tuple<T, T, T, T> zernike_polynomial_r_backward_backward(
    T gg_n, T gg_m, T gg_rho,
    T gradient, T n, T m, T rho) {

  T eps = T(1e-5);
  T gradient_gradient_output = T(0);
  T new_grad_n = T(0);
  T new_grad_m = T(0);
  T new_grad_rho = T(0);

  // Contribution from gg_n (second derivatives involving n)
  if (std::abs(gg_n) > T(1e-15)) {
    // d(grad_n)/d_gradient = d(gradient * dR/dn)/d_gradient = dR/dn
    auto [gn, gm, grho] = zernike_polynomial_r_backward(T(1), n, m, rho);
    gradient_gradient_output += gg_n * gn;

    // d(grad_n)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    T d2R_dn2 = (gn_plus - gn_minus) / (T(2) * eps);
    new_grad_n += gg_n * d2R_dn2;

    // d(grad_n)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    T d2R_dndm = (gn_m_plus - gn_m_minus) / (T(2) * eps);
    new_grad_m += gg_n * d2R_dndm;

    // d(grad_n)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    T d2R_dndrho = (gn_rho_plus - gn_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_n * d2R_dndrho;
  }

  // Contribution from gg_m (second derivatives involving m)
  if (std::abs(gg_m) > T(1e-15)) {
    // d(grad_m)/d_gradient = dR/dm
    auto [gn, gm, grho] = zernike_polynomial_r_backward(T(1), n, m, rho);
    gradient_gradient_output += gg_m * gm;

    // d(grad_m)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    T d2R_dmdn = (gm_plus - gm_minus) / (T(2) * eps);
    new_grad_n += gg_m * d2R_dmdn;

    // d(grad_m)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    T d2R_dm2 = (gm_m_plus - gm_m_minus) / (T(2) * eps);
    new_grad_m += gg_m * d2R_dm2;

    // d(grad_m)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    T d2R_dmdrho = (gm_rho_plus - gm_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_m * d2R_dmdrho;
  }

  // Contribution from gg_rho (second derivatives involving rho)
  if (std::abs(gg_rho) > T(1e-15)) {
    // d(grad_rho)/d_gradient = dR/drho
    auto [gn, gm, grho] = zernike_polynomial_r_backward(T(1), n, m, rho);
    gradient_gradient_output += gg_rho * grho;

    // d(grad_rho)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    T d2R_drhodn = (grho_plus - grho_minus) / (T(2) * eps);
    new_grad_n += gg_rho * d2R_drhodn;

    // d(grad_rho)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    T d2R_drhodm = (grho_m_plus - grho_m_minus) / (T(2) * eps);
    new_grad_m += gg_rho * d2R_drhodm;

    // d(grad_rho)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    T d2R_drho2 = (grho_rho_plus - grho_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_rho * d2R_drho2;
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_rho};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
zernike_polynomial_r_backward_backward(
    c10::complex<T> gg_n, c10::complex<T> gg_m, c10::complex<T> gg_rho,
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m, c10::complex<T> rho) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_n = zero;
  c10::complex<T> new_grad_m = zero;
  c10::complex<T> new_grad_rho = zero;

  // Contribution from gg_n
  if (std::abs(gg_n) > eps_val) {
    auto [gn, gm, grho] = zernike_polynomial_r_backward(one, n, m, rho);
    gradient_gradient_output += gg_n * std::conj(gn);

    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    c10::complex<T> d2R_dn2 = (gn_plus - gn_minus) / (two * eps);
    new_grad_n += gg_n * std::conj(d2R_dn2);

    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    c10::complex<T> d2R_dndm = (gn_m_plus - gn_m_minus) / (two * eps);
    new_grad_m += gg_n * std::conj(d2R_dndm);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    c10::complex<T> d2R_dndrho = (gn_rho_plus - gn_rho_minus) / (two * eps);
    new_grad_rho += gg_n * std::conj(d2R_dndrho);
  }

  // Contribution from gg_m
  if (std::abs(gg_m) > eps_val) {
    auto [gn, gm, grho] = zernike_polynomial_r_backward(one, n, m, rho);
    gradient_gradient_output += gg_m * std::conj(gm);

    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    c10::complex<T> d2R_dmdn = (gm_plus - gm_minus) / (two * eps);
    new_grad_n += gg_m * std::conj(d2R_dmdn);

    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    c10::complex<T> d2R_dm2 = (gm_m_plus - gm_m_minus) / (two * eps);
    new_grad_m += gg_m * std::conj(d2R_dm2);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    c10::complex<T> d2R_dmdrho = (gm_rho_plus - gm_rho_minus) / (two * eps);
    new_grad_rho += gg_m * std::conj(d2R_dmdrho);
  }

  // Contribution from gg_rho
  if (std::abs(gg_rho) > eps_val) {
    auto [gn, gm, grho] = zernike_polynomial_r_backward(one, n, m, rho);
    gradient_gradient_output += gg_rho * std::conj(grho);

    auto [gn_plus, gm_plus, grho_plus] = zernike_polynomial_r_backward(gradient, n + eps, m, rho);
    auto [gn_minus, gm_minus, grho_minus] = zernike_polynomial_r_backward(gradient, n - eps, m, rho);
    c10::complex<T> d2R_drhodn = (grho_plus - grho_minus) / (two * eps);
    new_grad_n += gg_rho * std::conj(d2R_drhodn);

    auto [gn_m_plus, gm_m_plus, grho_m_plus] = zernike_polynomial_r_backward(gradient, n, m + eps, rho);
    auto [gn_m_minus, gm_m_minus, grho_m_minus] = zernike_polynomial_r_backward(gradient, n, m - eps, rho);
    c10::complex<T> d2R_drhodm = (grho_m_plus - grho_m_minus) / (two * eps);
    new_grad_m += gg_rho * std::conj(d2R_drhodm);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus] = zernike_polynomial_r_backward(gradient, n, m, rho + eps);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus] = zernike_polynomial_r_backward(gradient, n, m, rho - eps);
    c10::complex<T> d2R_drho2 = (grho_rho_plus - grho_rho_minus) / (two * eps);
    new_grad_rho += gg_rho * std::conj(d2R_drho2);
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_rho};
}

} // namespace torchscience::kernel::special_functions
