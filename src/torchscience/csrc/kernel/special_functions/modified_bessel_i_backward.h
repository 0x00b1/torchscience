#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "modified_bessel_i.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dI_n/dn using finite differences
// The analytical formula is complex, involving integrals.
// For practical purposes, we use a numerical approximation.
template <typename T>
T modified_bessel_i_n_derivative(T n, T z) {
    const T eps = std::sqrt(modified_bessel_i_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T i_plus = modified_bessel_i(n + h, z);
    T i_minus = modified_bessel_i(n - h, z);

    return (i_plus - i_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> modified_bessel_i_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(modified_bessel_i_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> i_plus = modified_bessel_i(n + h, z);
    c10::complex<T> i_minus = modified_bessel_i(n - h, z);

    return (i_plus - i_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz I_n(z) = (I_{n-1}(z) + I_{n+1}(z)) / 2
// d/dn I_n(z) computed numerically
template <typename T>
std::tuple<T, T> modified_bessel_i_backward(T grad_output, T n, T z) {
    // Gradient w.r.t. z: d/dz I_n(z) = (I_{n-1}(z) + I_{n+1}(z)) / 2
    T i_nm1 = modified_bessel_i(n - T(1), z);
    T i_np1 = modified_bessel_i(n + T(1), z);
    T grad_z = grad_output * (i_nm1 + i_np1) / T(2);

    // Gradient w.r.t. n: computed numerically
    T di_dn = detail::modified_bessel_i_n_derivative(n, z);
    T grad_n = grad_output * di_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> modified_bessel_i_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // Gradient w.r.t. z: d/dz I_n(z) = (I_{n-1}(z) + I_{n+1}(z)) / 2
    c10::complex<T> i_nm1 = modified_bessel_i(n - one, z);
    c10::complex<T> i_np1 = modified_bessel_i(n + one, z);
    c10::complex<T> di_dz = (i_nm1 + i_np1) / two;

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(di_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> di_dn = detail::modified_bessel_i_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(di_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
