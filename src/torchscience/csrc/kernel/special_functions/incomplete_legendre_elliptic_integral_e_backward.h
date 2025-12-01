#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "incomplete_legendre_elliptic_integral_e.h"

namespace torchscience::kernel::special_functions {

// Backward pass for incomplete elliptic integral of the second kind E(phi, m)
//
// Uses numerical finite differences to compute gradients w.r.t. phi and m.
//
// The analytical derivatives are:
// dE/dphi = sqrt(1 - m*sin^2(phi))
// dE/dm = (E(phi, m) - F(phi, m)) / (2m)   where F is the incomplete integral of the first kind
//
// For simplicity and to avoid circular dependencies, we use numerical differentiation.

namespace detail {

template <typename T>
inline T incomplete_e_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float incomplete_e_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double incomplete_e_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> incomplete_legendre_elliptic_integral_e_backward(
    T gradient,
    T phi,
    T m
) {
    // Use central finite differences: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    T h = detail::incomplete_e_finite_diff_step<T>();

    // Gradient w.r.t. phi
    T f_phi_plus = incomplete_legendre_elliptic_integral_e(phi + h, m);
    T f_phi_minus = incomplete_legendre_elliptic_integral_e(phi - h, m);
    T derivative_phi = (f_phi_plus - f_phi_minus) / (T(2) * h);

    // Gradient w.r.t. m
    T f_m_plus = incomplete_legendre_elliptic_integral_e(phi, m + h);
    T f_m_minus = incomplete_legendre_elliptic_integral_e(phi, m - h);
    T derivative_m = (f_m_plus - f_m_minus) / (T(2) * h);

    return {gradient * derivative_phi, gradient * derivative_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> incomplete_legendre_elliptic_integral_e_backward(
    c10::complex<T> gradient,
    c10::complex<T> phi,
    c10::complex<T> m
) {
    // For complex, use finite differences in real direction
    // This follows the Wirtinger derivative convention
    T h = detail::incomplete_e_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. phi
    c10::complex<T> f_phi_plus = incomplete_legendre_elliptic_integral_e(phi + h_complex, m);
    c10::complex<T> f_phi_minus = incomplete_legendre_elliptic_integral_e(phi - h_complex, m);
    c10::complex<T> derivative_phi = (f_phi_plus - f_phi_minus) / (T(2) * h_complex);

    // Gradient w.r.t. m
    c10::complex<T> f_m_plus = incomplete_legendre_elliptic_integral_e(phi, m + h_complex);
    c10::complex<T> f_m_minus = incomplete_legendre_elliptic_integral_e(phi, m - h_complex);
    c10::complex<T> derivative_m = (f_m_plus - f_m_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return {
        gradient * std::conj(derivative_phi),
        gradient * std::conj(derivative_m)
    };
}

} // namespace torchscience::kernel::special_functions
