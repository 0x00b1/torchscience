#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "zernike_polynomial_z.h"
#include "zernike_polynomial_r.h"

namespace torchscience::kernel::special_functions {

// Backward pass for full Zernike polynomial Z_n^m(rho, theta)
//
// Z_n^m(rho, theta) = R_n^|m|(rho) * cos(m * theta)   if m >= 0
// Z_n^m(rho, theta) = R_n^|m|(rho) * sin(|m| * theta) if m < 0
//
// Derivatives:
// For m >= 0:
//   dZ/drho = dR/drho * cos(m * theta)
//   dZ/dtheta = -R * m * sin(m * theta)
//
// For m < 0:
//   dZ/drho = dR/drho * sin(|m| * theta)
//   dZ/dtheta = R * |m| * cos(|m| * theta)
//
// Derivatives with respect to n and m are computed via finite differences

template <typename T>
std::tuple<T, T, T, T> zernike_polynomial_z_backward(T gradient, T n, T m, T rho, T theta) {
  T eps = T(1e-7);
  T abs_m = std::abs(m);

  // Compute the radial part and its derivative
  T radial = zernike_polynomial_r(n, abs_m, rho);

  // dR/drho using finite differences
  T R_plus_rho = zernike_polynomial_r(n, abs_m, rho + eps);
  T R_minus_rho = zernike_polynomial_r(n, abs_m, rho - eps);
  T dR_drho = (R_plus_rho - R_minus_rho) / (T(2) * eps);

  T grad_rho, grad_theta;

  if (m >= T(0)) {
    // Z = R * cos(m * theta)
    T cos_m_theta = std::cos(m * theta);
    T sin_m_theta = std::sin(m * theta);

    grad_rho = dR_drho * cos_m_theta;
    grad_theta = -radial * m * sin_m_theta;
  } else {
    // Z = R * sin(|m| * theta)
    T cos_abs_m_theta = std::cos(abs_m * theta);
    T sin_abs_m_theta = std::sin(abs_m * theta);

    grad_rho = dR_drho * sin_abs_m_theta;
    grad_theta = radial * abs_m * cos_abs_m_theta;
  }

  // Gradient with respect to n using finite differences
  T Z_plus_n = zernike_polynomial_z(n + eps, m, rho, theta);
  T Z_minus_n = zernike_polynomial_z(n - eps, m, rho, theta);
  T grad_n = (Z_plus_n - Z_minus_n) / (T(2) * eps);

  // Gradient with respect to m using finite differences
  T Z_plus_m = zernike_polynomial_z(n, m + eps, rho, theta);
  T Z_minus_m = zernike_polynomial_z(n, m - eps, rho, theta);
  T grad_m = (Z_plus_m - Z_minus_m) / (T(2) * eps);

  return {gradient * grad_n, gradient * grad_m, gradient * grad_rho, gradient * grad_theta};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
zernike_polynomial_z_backward(
    c10::complex<T> gradient, c10::complex<T> n, c10::complex<T> m,
    c10::complex<T> rho, c10::complex<T> theta) {
  c10::complex<T> one(T(1), T(0));
  c10::complex<T> two(T(2), T(0));
  c10::complex<T> zero(T(0), T(0));
  T eps_val = T(1e-7);
  c10::complex<T> eps(eps_val, T(0));

  T m_real = m.real();
  T abs_m_real = std::abs(m_real);
  c10::complex<T> abs_m(abs_m_real, T(0));

  // Compute the radial part
  c10::complex<T> radial = zernike_polynomial_r(n, abs_m, rho);

  // dR/drho using finite differences
  c10::complex<T> R_plus_rho = zernike_polynomial_r(n, abs_m, rho + eps);
  c10::complex<T> R_minus_rho = zernike_polynomial_r(n, abs_m, rho - eps);
  c10::complex<T> dR_drho = (R_plus_rho - R_minus_rho) / (two * eps);

  c10::complex<T> grad_rho, grad_theta;

  if (m_real >= T(0)) {
    // Z = R * cos(m * theta)
    c10::complex<T> cos_m_theta = std::cos(m * theta);
    c10::complex<T> sin_m_theta = std::sin(m * theta);

    grad_rho = dR_drho * cos_m_theta;
    grad_theta = -radial * m * sin_m_theta;
  } else {
    // Z = R * sin(|m| * theta)
    c10::complex<T> cos_abs_m_theta = std::cos(abs_m * theta);
    c10::complex<T> sin_abs_m_theta = std::sin(abs_m * theta);

    grad_rho = dR_drho * sin_abs_m_theta;
    grad_theta = radial * abs_m * cos_abs_m_theta;
  }

  // Gradient with respect to n using finite differences
  c10::complex<T> Z_plus_n = zernike_polynomial_z(n + eps, m, rho, theta);
  c10::complex<T> Z_minus_n = zernike_polynomial_z(n - eps, m, rho, theta);
  c10::complex<T> grad_n = (Z_plus_n - Z_minus_n) / (two * eps);

  // Gradient with respect to m using finite differences
  c10::complex<T> Z_plus_m = zernike_polynomial_z(n, m + eps, rho, theta);
  c10::complex<T> Z_minus_m = zernike_polynomial_z(n, m - eps, rho, theta);
  c10::complex<T> grad_m = (Z_plus_m - Z_minus_m) / (two * eps);

  // For complex holomorphic functions, use Wirtinger derivative convention
  return {
    gradient * std::conj(grad_n),
    gradient * std::conj(grad_m),
    gradient * std::conj(grad_rho),
    gradient * std::conj(grad_theta)
  };
}

} // namespace torchscience::kernel::special_functions
