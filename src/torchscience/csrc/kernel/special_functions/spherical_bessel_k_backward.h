#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_k.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute d/dn k_n(z) using finite differences
// The analytical formula is complex, so we use numerical approximation
template <typename T>
T spherical_bessel_k_n_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_k_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T k_plus = spherical_bessel_k(n + h, z);
    T k_minus = spherical_bessel_k(n - h, z);

    return (k_plus - k_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_k_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> k_plus = spherical_bessel_k(n + h, z);
    c10::complex<T> k_minus = spherical_bessel_k(n - h, z);

    return (k_plus - k_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz k_n(z) = -k_{n-1}(z) - ((n+1)/z) * k_n(z)
// d/dn k_n(z) computed numerically
template <typename T>
std::tuple<T, T> spherical_bessel_k_backward(T grad_output, T n, T z) {
    const T eps = detail::spherical_bessel_k_eps<T>();

    // Gradient w.r.t. z: d/dz k_n(z) = -k_{n-1}(z) - ((n+1)/z) * k_n(z)
    T k_n = spherical_bessel_k(n, z);
    T k_nm1 = spherical_bessel_k(n - T(1), z);

    T grad_z;
    if (std::abs(z) < eps) {
        // At z=0, k_n is singular - gradient is also singular
        grad_z = -std::numeric_limits<T>::infinity();
    } else {
        grad_z = -k_nm1 - ((n + T(1)) / z) * k_n;
    }

    T grad_z_out = grad_output * grad_z;

    // Gradient w.r.t. n: computed numerically
    T dk_dn = detail::spherical_bessel_k_n_derivative(n, z);
    T grad_n = grad_output * dk_dn;

    return {grad_n, grad_z_out};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_k_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_k_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // Gradient w.r.t. z: d/dz k_n(z) = -k_{n-1}(z) - ((n+1)/z) * k_n(z)
    c10::complex<T> k_n = spherical_bessel_k(n, z);
    c10::complex<T> k_nm1 = spherical_bessel_k(n - one, z);

    c10::complex<T> dk_dz;
    if (std::abs(z) < eps) {
        // Handle z near 0 - singular
        dk_dz = c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    } else {
        dk_dz = -k_nm1 - ((n + one) / z) * k_n;
    }

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dk_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dk_dn = detail::spherical_bessel_k_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dk_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
