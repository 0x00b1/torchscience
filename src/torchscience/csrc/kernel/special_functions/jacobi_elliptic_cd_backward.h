#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "jacobi_elliptic_cd.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function cd(u, m)
//
// cd(u, m) = cn(u, m) / dn(u, m)
//
// Uses numerical finite differences for gradients.

namespace detail {

template <typename T>
inline T jacobi_cd_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float jacobi_cd_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double jacobi_cd_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_cd_backward(
    T gradient,
    T u,
    T m
) {
    // Use central finite differences: f'(x) = [f(x+h) - f(x-h)] / (2h)
    T h = detail::jacobi_cd_finite_diff_step<T>();

    // Gradient w.r.t. u
    T f_u_plus = jacobi_elliptic_cd(u + h, m);
    T f_u_minus = jacobi_elliptic_cd(u - h, m);
    T derivative_u = (f_u_plus - f_u_minus) / (T(2) * h);

    // Gradient w.r.t. m
    T f_m_plus = jacobi_elliptic_cd(u, m + h);
    T f_m_minus = jacobi_elliptic_cd(u, m - h);
    T derivative_m = (f_m_plus - f_m_minus) / (T(2) * h);

    return {gradient * derivative_u, gradient * derivative_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_cd_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    // For complex, use finite differences in real direction
    T h = detail::jacobi_cd_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. u
    c10::complex<T> f_u_plus = jacobi_elliptic_cd(u + h_complex, m);
    c10::complex<T> f_u_minus = jacobi_elliptic_cd(u - h_complex, m);
    c10::complex<T> derivative_u = (f_u_plus - f_u_minus) / (T(2) * h_complex);

    // Gradient w.r.t. m
    c10::complex<T> f_m_plus = jacobi_elliptic_cd(u, m + h_complex);
    c10::complex<T> f_m_minus = jacobi_elliptic_cd(u, m - h_complex);
    c10::complex<T> derivative_m = (f_m_plus - f_m_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return {
        gradient * std::conj(derivative_u),
        gradient * std::conj(derivative_m)
    };
}

} // namespace torchscience::kernel::special_functions
