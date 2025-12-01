#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "complete_legendre_elliptic_integral_pi.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> complete_legendre_elliptic_integral_pi_backward(
    T gradient,
    T n,
    T m
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for Pi gradients are complex
    // Pi(n, m) = R_F(0, 1-m, 1) + (n/3) * R_J(0, 1-m, 1, 1-n)

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    // Gradient w.r.t. n
    T pi_n_plus = complete_legendre_elliptic_integral_pi(n + eps, m);
    T pi_n_minus = complete_legendre_elliptic_integral_pi(n - eps, m);
    T dn = (pi_n_plus - pi_n_minus) / (T(2) * eps);

    // Gradient w.r.t. m
    T pi_m_plus = complete_legendre_elliptic_integral_pi(n, m + eps);
    T pi_m_minus = complete_legendre_elliptic_integral_pi(n, m - eps);
    T dm = (pi_m_plus - pi_m_minus) / (T(2) * eps);

    return {gradient * dn, gradient * dm};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
complete_legendre_elliptic_integral_pi_backward(
    c10::complex<T> gradient,
    c10::complex<T> n,
    c10::complex<T> m
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    // Gradient w.r.t. n
    c10::complex<T> pi_n_plus = complete_legendre_elliptic_integral_pi(n + eps_c, m);
    c10::complex<T> pi_n_minus = complete_legendre_elliptic_integral_pi(n - eps_c, m);
    c10::complex<T> dn = (pi_n_plus - pi_n_minus) / (T(2) * eps);

    // Gradient w.r.t. m
    c10::complex<T> pi_m_plus = complete_legendre_elliptic_integral_pi(n, m + eps_c);
    c10::complex<T> pi_m_minus = complete_legendre_elliptic_integral_pi(n, m - eps_c);
    c10::complex<T> dm = (pi_m_plus - pi_m_minus) / (T(2) * eps);

    return {gradient * std::conj(dn), gradient * std::conj(dm)};
}

} // namespace torchscience::kernel::special_functions
