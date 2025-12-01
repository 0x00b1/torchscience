#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_y.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute d/dn y_n(z) using finite differences
// The analytical formula is complex, so we use numerical approximation
template <typename T>
T spherical_bessel_y_n_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_y_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T y_plus = spherical_bessel_y(n + h, z);
    T y_minus = spherical_bessel_y(n - h, z);

    return (y_plus - y_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_y_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> y_plus = spherical_bessel_y(n + h, z);
    c10::complex<T> y_minus = spherical_bessel_y(n - h, z);

    return (y_plus - y_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz y_n(z) = (n/z) * y_n(z) - y_{n+1}(z)
// d/dn y_n(z) computed numerically
template <typename T>
std::tuple<T, T> spherical_bessel_y_backward(T grad_output, T n, T z) {
    const T eps = detail::spherical_bessel_y_eps<T>();

    // Gradient w.r.t. z: d/dz y_n(z) = (n/z) * y_n(z) - y_{n+1}(z)
    T y_n = spherical_bessel_y(n, z);
    T y_np1 = spherical_bessel_y(n + T(1), z);

    T grad_z;
    if (std::abs(z) < eps) {
        // At z=0, y_n is singular - gradient is also singular
        grad_z = -std::numeric_limits<T>::infinity();
    } else {
        grad_z = (n / z) * y_n - y_np1;
    }

    T grad_z_out = grad_output * grad_z;

    // Gradient w.r.t. n: computed numerically
    T dy_dn = detail::spherical_bessel_y_n_derivative(n, z);
    T grad_n = grad_output * dy_dn;

    return {grad_n, grad_z_out};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_y_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_y_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // Gradient w.r.t. z: d/dz y_n(z) = (n/z) * y_n(z) - y_{n+1}(z)
    c10::complex<T> y_n = spherical_bessel_y(n, z);
    c10::complex<T> y_np1 = spherical_bessel_y(n + one, z);

    c10::complex<T> dy_dz;
    if (std::abs(z) < eps) {
        // Handle z near 0 - singular
        dy_dz = c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    } else {
        dy_dz = (n / z) * y_n - y_np1;
    }

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(dy_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> dy_dn = detail::spherical_bessel_y_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(dy_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
