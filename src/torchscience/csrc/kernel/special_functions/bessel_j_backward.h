#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "bessel_j.h"
#include "digamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute ∂Jₙ/∂n using finite differences
// The analytical formula involves the integral representation and is complex.
// For practical purposes, we use a numerical approximation.
template <typename T>
T bessel_j_n_derivative(T n, T z) {
    const T eps = std::sqrt(bessel_j_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T j_plus = bessel_j(n + h, z);
    T j_minus = bessel_j(n - h, z);

    return (j_plus - j_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> bessel_j_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(bessel_j_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> j_plus = bessel_j(n + h, z);
    c10::complex<T> j_minus = bessel_j(n - h, z);

    return (j_plus - j_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// ∂/∂z Jₙ(z) = (Jₙ₋₁(z) - Jₙ₊₁(z)) / 2
// ∂/∂n Jₙ(z) computed numerically
template <typename T>
std::tuple<T, T> bessel_j_backward(T grad_output, T n, T z) {
    // Gradient w.r.t. z: d/dz J_n(z) = (J_{n-1}(z) - J_{n+1}(z)) / 2
    T j_nm1 = bessel_j(n - T(1), z);
    T j_np1 = bessel_j(n + T(1), z);
    T grad_z = grad_output * (j_nm1 - j_np1) / T(2);

    // Gradient w.r.t. n: computed numerically
    T dj_dn = detail::bessel_j_n_derivative(n, z);
    T grad_n = grad_output * dj_dn;

    return {grad_n, grad_z};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_j_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    // Gradient w.r.t. z: d/dz J_n(z) = (J_{n-1}(z) - J_{n+1}(z)) / 2
    c10::complex<T> j_nm1 = bessel_j(n - one, z);
    c10::complex<T> j_np1 = bessel_j(n + one, z);
    c10::complex<T> dj_dz = (j_nm1 - j_np1) / two;

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dj_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dj_dn = detail::bessel_j_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dj_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
