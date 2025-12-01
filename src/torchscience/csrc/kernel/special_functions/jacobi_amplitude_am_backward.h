#pragma once

#include <tuple>
#include <cmath>

#include "jacobi_amplitude_am.h"

namespace torchscience::kernel::special_functions {

/**
 * Backward pass for Jacobi amplitude function.
 *
 * Mathematical derivation:
 * Given: phi = am(u, m) where u = F(phi, m)
 *
 * Gradient with respect to u:
 *   d(am)/du = 1 / (dF/dphi) = 1 / (1/sqrt(1 - m*sin^2(phi))) = sqrt(1 - m*sin^2(phi)) = dn(u, m)
 *
 * Gradient with respect to m:
 *   Using implicit differentiation on u = F(phi, m):
 *   0 = (dF/dphi)(dphi/dm) + (dF/dm)
 *   dphi/dm = -(dF/dm) / (dF/dphi)
 *
 *   where dF/dphi = 1/sqrt(1 - m*sin^2(phi)) = 1/dn
 *   and dF/dm is the partial derivative of F with respect to m at constant phi.
 *
 *   For the incomplete elliptic integral:
 *   dF/dm = (E(phi, m) - (1-m)*F(phi, m)) / (2*m*(1-m)) - sin(phi)*cos(phi) / (2*(1-m)*sqrt(1-m*sin^2(phi)))
 *
 * For simplicity and numerical stability, we compute d(am)/dm using finite differences
 * when the analytical formula becomes numerically unstable (near m=0 or m=1).
 *
 * @param gradient The upstream gradient
 * @param u The argument
 * @param m The parameter
 * @return Tuple of (gradient_u, gradient_m)
 */
template <typename T>
std::tuple<T, T> jacobi_amplitude_am_backward(T gradient, T u, T m) {
    const T tolerance = detail::jacobi_amplitude_am_tolerance<T>();
    const T eps = T(1e-7);

    // Compute am(u, m)
    T phi = jacobi_amplitude_am(u, m);
    T sin_phi = std::sin(phi);
    T cos_phi = std::cos(phi);

    // Gradient w.r.t. u: d(am)/du = dn(u, m) = sqrt(1 - m * sin^2(phi))
    T sin2_phi = sin_phi * sin_phi;
    T dn = std::sqrt(std::max(T(1) - m * sin2_phi, T(0)));
    T grad_u = gradient * dn;

    // Gradient w.r.t. m: use numerical finite difference for stability
    // d(am)/dm approximately = (am(u, m + eps) - am(u, m - eps)) / (2 * eps)
    T grad_m;

    // Handle boundary cases for m
    if (m < eps) {
        // Near m = 0, use one-sided difference
        T phi_plus = jacobi_amplitude_am(u, m + eps);
        grad_m = gradient * (phi_plus - phi) / eps;
    } else if (m > T(1) - eps) {
        // Near m = 1, use one-sided difference
        T phi_minus = jacobi_amplitude_am(u, m - eps);
        grad_m = gradient * (phi - phi_minus) / eps;
    } else {
        // Central difference
        T phi_plus = jacobi_amplitude_am(u, m + eps);
        T phi_minus = jacobi_amplitude_am(u, m - eps);
        grad_m = gradient * (phi_plus - phi_minus) / (T(2) * eps);
    }

    return {grad_u, grad_m};
}

/**
 * Complex backward pass for Jacobi amplitude function.
 */
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_amplitude_am_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = T(1e-7);
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> eps_c(eps, T(0));

    // Compute am(u, m)
    c10::complex<T> phi = jacobi_amplitude_am(u, m);
    c10::complex<T> sin_phi = std::sin(phi);

    // Gradient w.r.t. u: d(am)/du = dn(u, m) = sqrt(1 - m * sin^2(phi))
    c10::complex<T> sin2_phi = sin_phi * sin_phi;
    c10::complex<T> dn = std::sqrt(one - m * sin2_phi);
    c10::complex<T> grad_u = gradient * dn;

    // Gradient w.r.t. m: numerical finite difference
    c10::complex<T> phi_plus = jacobi_amplitude_am(u, m + eps_c);
    c10::complex<T> phi_minus = jacobi_amplitude_am(u, m - eps_c);
    c10::complex<T> grad_m = gradient * (phi_plus - phi_minus) / (two * eps_c);

    return {grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
