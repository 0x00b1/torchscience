#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "incomplete_legendre_elliptic_integral_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> incomplete_legendre_elliptic_integral_pi_backward(
    T gradient,
    T n,
    T phi,
    T m
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for Pi gradients are complex
    // Pi(n, phi, m) = sin(phi) * R_F + (n/3) * sin^3(phi) * R_J

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    // Gradient w.r.t. n
    T pi_n_plus = incomplete_legendre_elliptic_integral_pi(n + eps, phi, m);
    T pi_n_minus = incomplete_legendre_elliptic_integral_pi(n - eps, phi, m);
    T dn = (pi_n_plus - pi_n_minus) / (T(2) * eps);

    // Gradient w.r.t. phi
    T pi_phi_plus = incomplete_legendre_elliptic_integral_pi(n, phi + eps, m);
    T pi_phi_minus = incomplete_legendre_elliptic_integral_pi(n, phi - eps, m);
    T dphi = (pi_phi_plus - pi_phi_minus) / (T(2) * eps);

    // Gradient w.r.t. m
    T pi_m_plus = incomplete_legendre_elliptic_integral_pi(n, phi, m + eps);
    T pi_m_minus = incomplete_legendre_elliptic_integral_pi(n, phi, m - eps);
    T dm = (pi_m_plus - pi_m_minus) / (T(2) * eps);

    return {gradient * dn, gradient * dphi, gradient * dm};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
incomplete_legendre_elliptic_integral_pi_backward(
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> phi,
    c10::complex<T> m
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    // Gradient w.r.t. n
    c10::complex<T> pi_n_plus = incomplete_legendre_elliptic_integral_pi(n + eps_c, phi, m);
    c10::complex<T> pi_n_minus = incomplete_legendre_elliptic_integral_pi(n - eps_c, phi, m);
    c10::complex<T> dn = (pi_n_plus - pi_n_minus) / (T(2) * eps);

    // Gradient w.r.t. phi
    c10::complex<T> pi_phi_plus = incomplete_legendre_elliptic_integral_pi(n, phi + eps_c, m);
    c10::complex<T> pi_phi_minus = incomplete_legendre_elliptic_integral_pi(n, phi - eps_c, m);
    c10::complex<T> dphi = (pi_phi_plus - pi_phi_minus) / (T(2) * eps);

    // Gradient w.r.t. m
    c10::complex<T> pi_m_plus = incomplete_legendre_elliptic_integral_pi(n, phi, m + eps_c);
    c10::complex<T> pi_m_minus = incomplete_legendre_elliptic_integral_pi(n, phi, m - eps_c);
    c10::complex<T> dm = (pi_m_plus - pi_m_minus) / (T(2) * eps);

    return {gradient * std::conj(dn), gradient * std::conj(dphi), gradient * std::conj(dm)};
}

} // namespace torchscience::kernel::special_functions
