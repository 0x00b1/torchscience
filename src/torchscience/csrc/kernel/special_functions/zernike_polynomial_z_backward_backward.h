#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "zernike_polynomial_z.h"
#include "zernike_polynomial_z_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for full Zernike polynomial Z_n^m(rho, theta)
//
// Uses finite differences for second derivatives

template <typename T>
std::tuple<T, T, T, T, T> zernike_polynomial_z_backward_backward(
    T gg_n, T gg_m, T gg_rho, T gg_theta,
    T gradient, T n, T m, T rho, T theta) {

  T eps = T(1e-5);
  T gradient_gradient_output = T(0);
  T new_grad_n = T(0);
  T new_grad_m = T(0);
  T new_grad_rho = T(0);
  T new_grad_theta = T(0);

  // Contribution from gg_n
  if (std::abs(gg_n) > T(1e-15)) {
    // d(grad_n)/d_gradient = dZ/dn
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(T(1), n, m, rho, theta);
    gradient_gradient_output += gg_n * gn;

    // d(grad_n)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    T d2Z_dn2 = (gn_plus - gn_minus) / (T(2) * eps);
    new_grad_n += gg_n * d2Z_dn2;

    // d(grad_n)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    T d2Z_dndm = (gn_m_plus - gn_m_minus) / (T(2) * eps);
    new_grad_m += gg_n * d2Z_dndm;

    // d(grad_n)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    T d2Z_dndrho = (gn_rho_plus - gn_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_n * d2Z_dndrho;

    // d(grad_n)/dtheta using finite difference
    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    T d2Z_dndtheta = (gn_theta_plus - gn_theta_minus) / (T(2) * eps);
    new_grad_theta += gg_n * d2Z_dndtheta;
  }

  // Contribution from gg_m
  if (std::abs(gg_m) > T(1e-15)) {
    // d(grad_m)/d_gradient = dZ/dm
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(T(1), n, m, rho, theta);
    gradient_gradient_output += gg_m * gm;

    // d(grad_m)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    T d2Z_dmdn = (gm_plus - gm_minus) / (T(2) * eps);
    new_grad_n += gg_m * d2Z_dmdn;

    // d(grad_m)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    T d2Z_dm2 = (gm_m_plus - gm_m_minus) / (T(2) * eps);
    new_grad_m += gg_m * d2Z_dm2;

    // d(grad_m)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    T d2Z_dmdrho = (gm_rho_plus - gm_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_m * d2Z_dmdrho;

    // d(grad_m)/dtheta using finite difference
    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    T d2Z_dmdtheta = (gm_theta_plus - gm_theta_minus) / (T(2) * eps);
    new_grad_theta += gg_m * d2Z_dmdtheta;
  }

  // Contribution from gg_rho
  if (std::abs(gg_rho) > T(1e-15)) {
    // d(grad_rho)/d_gradient = dZ/drho
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(T(1), n, m, rho, theta);
    gradient_gradient_output += gg_rho * grho;

    // d(grad_rho)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    T d2Z_drhodn = (grho_plus - grho_minus) / (T(2) * eps);
    new_grad_n += gg_rho * d2Z_drhodn;

    // d(grad_rho)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    T d2Z_drhodm = (grho_m_plus - grho_m_minus) / (T(2) * eps);
    new_grad_m += gg_rho * d2Z_drhodm;

    // d(grad_rho)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    T d2Z_drho2 = (grho_rho_plus - grho_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_rho * d2Z_drho2;

    // d(grad_rho)/dtheta using finite difference
    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    T d2Z_drhodtheta = (grho_theta_plus - grho_theta_minus) / (T(2) * eps);
    new_grad_theta += gg_rho * d2Z_drhodtheta;
  }

  // Contribution from gg_theta
  if (std::abs(gg_theta) > T(1e-15)) {
    // d(grad_theta)/d_gradient = dZ/dtheta
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(T(1), n, m, rho, theta);
    gradient_gradient_output += gg_theta * gtheta;

    // d(grad_theta)/dn using finite difference
    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    T d2Z_dthetadn = (gtheta_plus - gtheta_minus) / (T(2) * eps);
    new_grad_n += gg_theta * d2Z_dthetadn;

    // d(grad_theta)/dm using finite difference
    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    T d2Z_dthetadm = (gtheta_m_plus - gtheta_m_minus) / (T(2) * eps);
    new_grad_m += gg_theta * d2Z_dthetadm;

    // d(grad_theta)/drho using finite difference
    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    T d2Z_dthetadrho = (gtheta_rho_plus - gtheta_rho_minus) / (T(2) * eps);
    new_grad_rho += gg_theta * d2Z_dthetadrho;

    // d(grad_theta)/dtheta using finite difference
    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    T d2Z_dtheta2 = (gtheta_theta_plus - gtheta_theta_minus) / (T(2) * eps);
    new_grad_theta += gg_theta * d2Z_dtheta2;
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_rho, new_grad_theta};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
zernike_polynomial_z_backward_backward(
    c10::complex<T> gg_n, c10::complex<T> gg_m, c10::complex<T> gg_rho, c10::complex<T> gg_theta,
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m,
    c10::complex<T> rho, c10::complex<T> theta) {

  T eps_val = T(1e-5);
  c10::complex<T> eps(eps_val, T(0));
  c10::complex<T> zero(T(0), T(0));
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));

  c10::complex<T> gradient_gradient_output = zero;
  c10::complex<T> new_grad_n = zero;
  c10::complex<T> new_grad_m = zero;
  c10::complex<T> new_grad_rho = zero;
  c10::complex<T> new_grad_theta = zero;

  // Contribution from gg_n
  if (std::abs(gg_n) > eps_val) {
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(one, n, m, rho, theta);
    gradient_gradient_output += gg_n * std::conj(gn);

    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    c10::complex<T> d2Z_dn2 = (gn_plus - gn_minus) / (two * eps);
    new_grad_n += gg_n * std::conj(d2Z_dn2);

    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    c10::complex<T> d2Z_dndm = (gn_m_plus - gn_m_minus) / (two * eps);
    new_grad_m += gg_n * std::conj(d2Z_dndm);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    c10::complex<T> d2Z_dndrho = (gn_rho_plus - gn_rho_minus) / (two * eps);
    new_grad_rho += gg_n * std::conj(d2Z_dndrho);

    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    c10::complex<T> d2Z_dndtheta = (gn_theta_plus - gn_theta_minus) / (two * eps);
    new_grad_theta += gg_n * std::conj(d2Z_dndtheta);
  }

  // Contribution from gg_m
  if (std::abs(gg_m) > eps_val) {
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(one, n, m, rho, theta);
    gradient_gradient_output += gg_m * std::conj(gm);

    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    c10::complex<T> d2Z_dmdn = (gm_plus - gm_minus) / (two * eps);
    new_grad_n += gg_m * std::conj(d2Z_dmdn);

    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    c10::complex<T> d2Z_dm2 = (gm_m_plus - gm_m_minus) / (two * eps);
    new_grad_m += gg_m * std::conj(d2Z_dm2);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    c10::complex<T> d2Z_dmdrho = (gm_rho_plus - gm_rho_minus) / (two * eps);
    new_grad_rho += gg_m * std::conj(d2Z_dmdrho);

    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    c10::complex<T> d2Z_dmdtheta = (gm_theta_plus - gm_theta_minus) / (two * eps);
    new_grad_theta += gg_m * std::conj(d2Z_dmdtheta);
  }

  // Contribution from gg_rho
  if (std::abs(gg_rho) > eps_val) {
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(one, n, m, rho, theta);
    gradient_gradient_output += gg_rho * std::conj(grho);

    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    c10::complex<T> d2Z_drhodn = (grho_plus - grho_minus) / (two * eps);
    new_grad_n += gg_rho * std::conj(d2Z_drhodn);

    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    c10::complex<T> d2Z_drhodm = (grho_m_plus - grho_m_minus) / (two * eps);
    new_grad_m += gg_rho * std::conj(d2Z_drhodm);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    c10::complex<T> d2Z_drho2 = (grho_rho_plus - grho_rho_minus) / (two * eps);
    new_grad_rho += gg_rho * std::conj(d2Z_drho2);

    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    c10::complex<T> d2Z_drhodtheta = (grho_theta_plus - grho_theta_minus) / (two * eps);
    new_grad_theta += gg_rho * std::conj(d2Z_drhodtheta);
  }

  // Contribution from gg_theta
  if (std::abs(gg_theta) > eps_val) {
    auto [gn, gm, grho, gtheta] = zernike_polynomial_z_backward(one, n, m, rho, theta);
    gradient_gradient_output += gg_theta * std::conj(gtheta);

    auto [gn_plus, gm_plus, grho_plus, gtheta_plus] = zernike_polynomial_z_backward(gradient, n + eps, m, rho, theta);
    auto [gn_minus, gm_minus, grho_minus, gtheta_minus] = zernike_polynomial_z_backward(gradient, n - eps, m, rho, theta);
    c10::complex<T> d2Z_dthetadn = (gtheta_plus - gtheta_minus) / (two * eps);
    new_grad_n += gg_theta * std::conj(d2Z_dthetadn);

    auto [gn_m_plus, gm_m_plus, grho_m_plus, gtheta_m_plus] = zernike_polynomial_z_backward(gradient, n, m + eps, rho, theta);
    auto [gn_m_minus, gm_m_minus, grho_m_minus, gtheta_m_minus] = zernike_polynomial_z_backward(gradient, n, m - eps, rho, theta);
    c10::complex<T> d2Z_dthetadm = (gtheta_m_plus - gtheta_m_minus) / (two * eps);
    new_grad_m += gg_theta * std::conj(d2Z_dthetadm);

    auto [gn_rho_plus, gm_rho_plus, grho_rho_plus, gtheta_rho_plus] = zernike_polynomial_z_backward(gradient, n, m, rho + eps, theta);
    auto [gn_rho_minus, gm_rho_minus, grho_rho_minus, gtheta_rho_minus] = zernike_polynomial_z_backward(gradient, n, m, rho - eps, theta);
    c10::complex<T> d2Z_dthetadrho = (gtheta_rho_plus - gtheta_rho_minus) / (two * eps);
    new_grad_rho += gg_theta * std::conj(d2Z_dthetadrho);

    auto [gn_theta_plus, gm_theta_plus, grho_theta_plus, gtheta_theta_plus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta + eps);
    auto [gn_theta_minus, gm_theta_minus, grho_theta_minus, gtheta_theta_minus] = zernike_polynomial_z_backward(gradient, n, m, rho, theta - eps);
    c10::complex<T> d2Z_dtheta2 = (gtheta_theta_plus - gtheta_theta_minus) / (two * eps);
    new_grad_theta += gg_theta * std::conj(d2Z_dtheta2);
  }

  return {gradient_gradient_output, new_grad_n, new_grad_m, new_grad_rho, new_grad_theta};
}

} // namespace torchscience::kernel::special_functions
