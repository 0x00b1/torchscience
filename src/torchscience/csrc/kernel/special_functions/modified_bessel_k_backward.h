#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "modified_bessel_k.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute dK_n/dn using finite differences
// The analytical formula is complex, involving integrals.
// For practical purposes, we use a numerical approximation.
template <typename T>
T modified_bessel_k_n_derivative(T n, T z) {
    const T eps = std::sqrt(modified_bessel_k_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T k_plus = modified_bessel_k(n + h, z);
    T k_minus = modified_bessel_k(n - h, z);

    return (k_plus - k_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> modified_bessel_k_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(modified_bessel_k_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> k_plus = modified_bessel_k(n + h, z);
    c10::complex<T> k_minus = modified_bessel_k(n - h, z);

    return (k_plus - k_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz K_n(z) = -(K_{n-1}(z) + K_{n+1}(z)) / 2
// d/dn K_n(z) computed numerically
template <typename T>
std::tuple<T, T> modified_bessel_k_backward(T grad_output, T n, T z) {
    // Gradient w.r.t. z: d/dz K_n(z) = -(K_{n-1}(z) + K_{n+1}(z)) / 2
    T k_nm1 = modified_bessel_k(n - T(1), z);
    T k_np1 = modified_bessel_k(n + T(1), z);
    T grad_z = grad_output * (-(k_nm1 + k_np1) / T(2));

    // Gradient w.r.t. n: computed numerically
    T dk_dn = detail::modified_bessel_k_n_derivative(n, z);
    T grad_n = grad_output * dk_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> modified_bessel_k_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // Gradient w.r.t. z: d/dz K_n(z) = -(K_{n-1}(z) + K_{n+1}(z)) / 2
    c10::complex<T> k_nm1 = modified_bessel_k(n - one, z);
    c10::complex<T> k_np1 = modified_bessel_k(n + one, z);
    c10::complex<T> dk_dz = -(k_nm1 + k_np1) / two;

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dk_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dk_dn = detail::modified_bessel_k_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dk_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
