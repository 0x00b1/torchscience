#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_i.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Compute d/dn i_n(z) using finite differences
// The analytical formula is complex, so we use numerical approximation
template <typename T>
T spherical_bessel_i_n_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_i_eps<T>());

    // Central difference approximation
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T i_plus = spherical_bessel_i(n + h, z);
    T i_minus = spherical_bessel_i(n - h, z);

    return (i_plus - i_minus) / (T(2) * h);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_i_n_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_i_eps<T>());
    const c10::complex<T> h_c(eps, T(0));

    // Scale h based on |n|
    T n_mag = std::abs(n);
    c10::complex<T> h = (n_mag > T(1)) ? h_c * c10::complex<T>(n_mag, T(0)) : h_c;

    c10::complex<T> i_plus = spherical_bessel_i(n + h, z);
    c10::complex<T> i_minus = spherical_bessel_i(n - h, z);

    return (i_plus - i_minus) / (c10::complex<T>(T(2), T(0)) * h);
}

} // namespace detail

// Real backward: returns (grad_n, grad_z)
// d/dz i_n(z) = (n/z) * i_n(z) + i_{n+1}(z)  (note: + sign, unlike j_n)
// d/dn i_n(z) computed numerically
template <typename T>
std::tuple<T, T> spherical_bessel_i_backward(T grad_output, T n, T z) {
    const T eps = detail::spherical_bessel_i_eps<T>();

    // Gradient w.r.t. z: d/dz i_n(z) = (n/z) * i_n(z) + i_{n+1}(z)
    // Alternative formula: d/dz i_n(z) = i_{n-1}(z) - (n+1)/z * i_n(z)
    // Using the first formula for consistency

    T i_n = spherical_bessel_i(n, z);
    T i_np1 = spherical_bessel_i(n + T(1), z);

    T grad_z;
    if (std::abs(z) < eps) {
        // At z=0, use series expansion
        // i_n(z) ~ (z^n) / (2n+1)!! for small z
        // d/dz i_n(z) ~ n * z^{n-1} / (2n+1)!! for n > 0
        // For n=0: d/dz i_0(0) = 0
        if (std::abs(n) < eps) {
            grad_z = T(0);  // i_0'(0) = 0
        } else if (n > T(0)) {
            // Limit depends on n; for n=1, i_1'(0) = 1/3
            if (std::abs(n - T(1)) < eps) {
                grad_z = T(1) / T(3);
            } else {
                grad_z = T(0);  // For n > 1, derivative at 0 is 0
            }
        } else {
            grad_z = std::numeric_limits<T>::quiet_NaN();
        }
    } else {
        grad_z = (n / z) * i_n + i_np1;
    }

    T grad_z_out = grad_output * grad_z;

    // Gradient w.r.t. n: computed numerically
    T di_dn = detail::spherical_bessel_i_n_derivative(n, z);
    T grad_n = grad_output * di_dn;

    return {grad_n, grad_z_out};
}

// Complex backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_i_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_i_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // Gradient w.r.t. z: d/dz i_n(z) = (n/z) * i_n(z) + i_{n+1}(z)
    c10::complex<T> i_n = spherical_bessel_i(n, z);
    c10::complex<T> i_np1 = spherical_bessel_i(n + one, z);

    c10::complex<T> di_dz;
    if (std::abs(z) < eps) {
        // Handle z near 0
        if (std::abs(n) < eps) {
            di_dz = c10::complex<T>(T(0), T(0));
        } else if (n.real() > T(0)) {
            if (std::abs(n - one) < eps) {
                di_dz = c10::complex<T>(T(1) / T(3), T(0));
            } else {
                di_dz = c10::complex<T>(T(0), T(0));
            }
        } else {
            di_dz = c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
        }
    } else {
        di_dz = (n / z) * i_n + i_np1;
    }

    // For complex gradients, we use the conjugate (Wirtinger derivative)
    c10::complex<T> grad_z = grad_output * std::conj(di_dz);

    // Gradient w.r.t. n: computed numerically
    c10::complex<T> di_dn = detail::spherical_bessel_i_n_derivative(n, z);
    c10::complex<T> grad_n = grad_output * std::conj(di_dn);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
