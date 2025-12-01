#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "jacobi_elliptic_sn.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function sn(u, m)
//
// Analytical derivatives:
// dsn/du = cn(u, m) * dn(u, m)
// dsn/dm = (1/(2m)) * (E(am(u,m), m) - (1-m)*u*cn(u,m)*dn(u,m)/sn(u,m) - cn(u,m)*dn(u,m)*u)
//          - more complex expression involving other Jacobi functions
//
// For simplicity and to avoid circular dependencies with cn and dn,
// we use numerical finite differences.

namespace detail {

template <typename T>
inline T jacobi_sn_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float jacobi_sn_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double jacobi_sn_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_sn_backward(
    T gradient,
    T u,
    T m
) {
    // Use central finite differences: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    T h = detail::jacobi_sn_finite_diff_step<T>();

    // Gradient w.r.t. u
    T f_u_plus = jacobi_elliptic_sn(u + h, m);
    T f_u_minus = jacobi_elliptic_sn(u - h, m);
    T derivative_u = (f_u_plus - f_u_minus) / (T(2) * h);

    // Gradient w.r.t. m
    // Need to be careful near m = 0 and m = 1 boundaries
    T m_plus = m + h;
    T m_minus = m - h;

    // Clamp m values to avoid issues at boundaries if needed
    // For now, allow any m value as the forward pass handles edge cases

    T f_m_plus = jacobi_elliptic_sn(u, m_plus);
    T f_m_minus = jacobi_elliptic_sn(u, m_minus);
    T derivative_m = (f_m_plus - f_m_minus) / (T(2) * h);

    return {gradient * derivative_u, gradient * derivative_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_sn_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    // For complex, use finite differences in real direction
    // This follows the Wirtinger derivative convention
    T h = detail::jacobi_sn_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. u
    c10::complex<T> f_u_plus = jacobi_elliptic_sn(u + h_complex, m);
    c10::complex<T> f_u_minus = jacobi_elliptic_sn(u - h_complex, m);
    c10::complex<T> derivative_u = (f_u_plus - f_u_minus) / (T(2) * h_complex);

    // Gradient w.r.t. m
    c10::complex<T> f_m_plus = jacobi_elliptic_sn(u, m + h_complex);
    c10::complex<T> f_m_minus = jacobi_elliptic_sn(u, m - h_complex);
    c10::complex<T> derivative_m = (f_m_plus - f_m_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return {
        gradient * std::conj(derivative_u),
        gradient * std::conj(derivative_m)
    };
}

} // namespace torchscience::kernel::special_functions
