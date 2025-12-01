#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "inverse_jacobi_elliptic_cd.h"

namespace torchscience::kernel::special_functions {

// Backward pass for inverse Jacobi elliptic function arccd(x, m)
//
// arccd(x, m) = u such that cd(u, m) = x
//
// Uses numerical finite differences for gradients.

namespace detail {

template <typename T>
inline T inverse_jacobi_cd_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float inverse_jacobi_cd_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double inverse_jacobi_cd_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> inverse_jacobi_elliptic_cd_backward(
    T gradient,
    T x,
    T m
) {
    // Use central finite differences: f'(x) = [f(x+h) - f(x-h)] / (2h)
    T h = detail::inverse_jacobi_cd_finite_diff_step<T>();

    // Gradient w.r.t. x
    T f_x_plus = inverse_jacobi_elliptic_cd(x + h, m);
    T f_x_minus = inverse_jacobi_elliptic_cd(x - h, m);
    T derivative_x = (f_x_plus - f_x_minus) / (T(2) * h);

    // Gradient w.r.t. m
    T f_m_plus = inverse_jacobi_elliptic_cd(x, m + h);
    T f_m_minus = inverse_jacobi_elliptic_cd(x, m - h);
    T derivative_m = (f_m_plus - f_m_minus) / (T(2) * h);

    return {gradient * derivative_x, gradient * derivative_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> inverse_jacobi_elliptic_cd_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> m
) {
    // For complex, use finite differences in real direction
    T h = detail::inverse_jacobi_cd_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. x
    c10::complex<T> f_x_plus = inverse_jacobi_elliptic_cd(x + h_complex, m);
    c10::complex<T> f_x_minus = inverse_jacobi_elliptic_cd(x - h_complex, m);
    c10::complex<T> derivative_x = (f_x_plus - f_x_minus) / (T(2) * h_complex);

    // Gradient w.r.t. m
    c10::complex<T> f_m_plus = inverse_jacobi_elliptic_cd(x, m + h_complex);
    c10::complex<T> f_m_minus = inverse_jacobi_elliptic_cd(x, m - h_complex);
    c10::complex<T> derivative_m = (f_m_plus - f_m_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return {
        gradient * std::conj(derivative_x),
        gradient * std::conj(derivative_m)
    };
}

} // namespace torchscience::kernel::special_functions
