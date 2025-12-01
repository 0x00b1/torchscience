#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "spherical_bessel_y.h"
#include "spherical_bessel_y_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative w.r.t. z: d^2/dz^2 y_n(z)
// Using: d/dz[(n/z)*y_n - y_{n+1}] = -n/z^2 * y_n + n/z * dy_n/dz - dy_{n+1}/dz
// where dy_n/dz = (n/z)*y_n - y_{n+1}
// and dy_{n+1}/dz = ((n+1)/z)*y_{n+1} - y_{n+2}
template <typename T>
T spherical_bessel_y_zz_derivative(T n, T z) {
    const T eps = spherical_bessel_y_eps<T>();

    if (std::abs(z) < eps) {
        // For small z, y_n and its derivatives are singular
        return -std::numeric_limits<T>::infinity();
    }

    T y_n = spherical_bessel_y(n, z);
    T y_np1 = spherical_bessel_y(n + T(1), z);
    T y_np2 = spherical_bessel_y(n + T(2), z);

    // d/dz y_n = (n/z) * y_n - y_{n+1}
    T dy_n_dz = (n / z) * y_n - y_np1;

    // d/dz y_{n+1} = ((n+1)/z) * y_{n+1} - y_{n+2}
    T dy_np1_dz = ((n + T(1)) / z) * y_np1 - y_np2;

    // d^2/dz^2 y_n = d/dz[(n/z)*y_n - y_{n+1}]
    //             = -n/z^2 * y_n + n/z * dy_n/dz - dy_{n+1}/dz
    T z2 = z * z;
    return -n / z2 * y_n + (n / z) * dy_n_dz - dy_np1_dz;
}

// Mixed second derivative d^2/(dn dz) y_n(z) computed numerically
template <typename T>
T spherical_bessel_y_nz_derivative(T n, T z) {
    const T eps = std::sqrt(spherical_bessel_y_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // Compute d/dz y_{n+h} and d/dz y_{n-h}
    T y_p = spherical_bessel_y(n + h, z);
    T y_p_p1 = spherical_bessel_y(n + h + T(1), z);
    T dy_dz_plus = ((n + h) / z) * y_p - y_p_p1;

    T y_m = spherical_bessel_y(n - h, z);
    T y_m_p1 = spherical_bessel_y(n - h + T(1), z);
    T dy_dz_minus = ((n - h) / z) * y_m - y_m_p1;

    return (dy_dz_plus - dy_dz_minus) / (T(2) * h);
}

// Second derivative w.r.t. n: d^2/dn^2 y_n(z) computed numerically
template <typename T>
T spherical_bessel_y_nn_derivative(T n, T z) {
    const T eps = std::cbrt(spherical_bessel_y_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T y_plus = spherical_bessel_y(n + h, z);
    T y_center = spherical_bessel_y(n, z);
    T y_minus = spherical_bessel_y(n - h, z);

    return (y_plus - T(2) * y_center + y_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> spherical_bessel_y_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = spherical_bessel_y_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));

    if (std::abs(z) < eps) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> y_n = spherical_bessel_y(n, z);
    c10::complex<T> y_np1 = spherical_bessel_y(n + one, z);
    c10::complex<T> y_np2 = spherical_bessel_y(n + two, z);

    c10::complex<T> dy_n_dz = (n / z) * y_n - y_np1;
    c10::complex<T> dy_np1_dz = ((n + one) / z) * y_np1 - y_np2;

    c10::complex<T> z2 = z * z;
    return -n / z2 * y_n + (n / z) * dy_n_dz - dy_np1_dz;
}

template <typename T>
c10::complex<T> spherical_bessel_y_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(spherical_bessel_y_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> y_p = spherical_bessel_y(n + h, z);
    c10::complex<T> y_p_p1 = spherical_bessel_y(n + h + one, z);
    c10::complex<T> dy_dz_plus = ((n + h) / z) * y_p - y_p_p1;

    c10::complex<T> y_m = spherical_bessel_y(n - h, z);
    c10::complex<T> y_m_p1 = spherical_bessel_y(n - h + one, z);
    c10::complex<T> dy_dz_minus = ((n - h) / z) * y_m - y_m_p1;

    return (dy_dz_plus - dy_dz_minus) / (two * h);
}

template <typename T>
c10::complex<T> spherical_bessel_y_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(spherical_bessel_y_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> y_plus = spherical_bessel_y(n + h, z);
    c10::complex<T> y_center = spherical_bessel_y(n, z);
    c10::complex<T> y_minus = spherical_bessel_y(n - h, z);

    return (y_plus - two * y_center + y_minus) / (h * h);
}

} // namespace detail

// Real backward_backward: returns (grad_grad_output, grad_n, grad_z)
// Computes gradients of the backward pass w.r.t. (grad_output, n, z)
// given upstream gradients (gg_n, gg_z) for the outputs (grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> spherical_bessel_y_backward_backward(
    T gg_n,       // upstream gradient for grad_n output
    T gg_z,       // upstream gradient for grad_z output
    T grad_output,
    T n,
    T z
) {
    const T eps = detail::spherical_bessel_y_eps<T>();

    // Forward backward computes:
    // grad_n = grad_output * dy/dn
    // grad_z = grad_output * dy/dz

    // First derivatives
    T y_n = spherical_bessel_y(n, z);
    T y_np1 = spherical_bessel_y(n + T(1), z);

    T dy_dz;
    if (std::abs(z) < eps) {
        dy_dz = -std::numeric_limits<T>::infinity();
    } else {
        dy_dz = (n / z) * y_n - y_np1;
    }

    T dy_dn = detail::spherical_bessel_y_n_derivative(n, z);

    // Second derivatives
    T d2y_dz2 = detail::spherical_bessel_y_zz_derivative(n, z);
    T d2y_dn2 = detail::spherical_bessel_y_nn_derivative(n, z);
    T d2y_dndz = detail::spherical_bessel_y_nz_derivative(n, z);

    // Accumulate gradients
    // grad_grad_output = gg_n * dy/dn + gg_z * dy/dz
    T grad_grad_output = gg_n * dy_dn + gg_z * dy_dz;

    // grad_n = gg_n * grad_output * d^2y/dn^2 + gg_z * grad_output * d^2y/(dn dz)
    T grad_n = grad_output * (gg_n * d2y_dn2 + gg_z * d2y_dndz);

    // grad_z = gg_n * grad_output * d^2y/(dn dz) + gg_z * grad_output * d^2y/dz^2
    T grad_z = grad_output * (gg_n * d2y_dndz + gg_z * d2y_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> spherical_bessel_y_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const T eps = detail::spherical_bessel_y_eps<T>();
    const c10::complex<T> one(T(1), T(0));

    // First derivatives
    c10::complex<T> y_n = spherical_bessel_y(n, z);
    c10::complex<T> y_np1 = spherical_bessel_y(n + one, z);

    c10::complex<T> dy_dz;
    if (std::abs(z) < eps) {
        dy_dz = c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    } else {
        dy_dz = (n / z) * y_n - y_np1;
    }

    c10::complex<T> dy_dn = detail::spherical_bessel_y_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2y_dz2 = detail::spherical_bessel_y_zz_derivative(n, z);
    c10::complex<T> d2y_dn2 = detail::spherical_bessel_y_nn_derivative(n, z);
    c10::complex<T> d2y_dndz = detail::spherical_bessel_y_nz_derivative(n, z);

    // Accumulate gradients (using conjugates for Wirtinger derivatives)
    c10::complex<T> grad_grad_output = gg_n * std::conj(dy_dn) + gg_z * std::conj(dy_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2y_dn2) + gg_z * std::conj(d2y_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2y_dndz) + gg_z * std::conj(d2y_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
