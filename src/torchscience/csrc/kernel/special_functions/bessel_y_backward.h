#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "bessel_y.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dY_n/dn using finite differences
// The analytical formula is complex, involving integrals and digamma functions.
// For practical purposes, we use a numerical approximation.
template <typename T>
T bessel_y_n_derivative(T n, T z) {
    const T eps = std::sqrt(bessel_y_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T y_plus = bessel_y(n + h, z);
    T y_minus = bessel_y(n - h, z);

    return (y_plus - y_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> bessel_y_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(bessel_y_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> y_plus = bessel_y(n + h, z);
    c10::complex<T> y_minus = bessel_y(n - h, z);

    return (y_plus - y_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz Y_n(z) = (Y_{n-1}(z) - Y_{n+1}(z)) / 2
// d/dn Y_n(z) computed numerically
template <typename T>
std::tuple<T, T> bessel_y_backward(T grad_output, T n, T z) {
    // Gradient w.r.t. z: d/dz Y_n(z) = (Y_{n-1}(z) - Y_{n+1}(z)) / 2
    T y_nm1 = bessel_y(n - T(1), z);
    T y_np1 = bessel_y(n + T(1), z);
    T grad_z = grad_output * (y_nm1 - y_np1) / T(2);

    // Gradient w.r.t. n: computed numerically
    T dy_dn = detail::bessel_y_n_derivative(n, z);
    T grad_n = grad_output * dy_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_y_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // Gradient w.r.t. z: d/dz Y_n(z) = (Y_{n-1}(z) - Y_{n+1}(z)) / 2
    c10::complex<T> y_nm1 = bessel_y(n - one, z);
    c10::complex<T> y_np1 = bessel_y(n + one, z);
    c10::complex<T> dy_dz = (y_nm1 - y_np1) / two;

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dy_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dy_dn = detail::bessel_y_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dy_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
