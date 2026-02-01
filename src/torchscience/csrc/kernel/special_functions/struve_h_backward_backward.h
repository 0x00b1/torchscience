#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "struve_h.h"
#include "struve_h_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative d^2 H_n / dz^2
// Using dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
// d^2H_n/dz^2 = dH_{n-1}/dz - d/dz[(n/z) * H_n(z)]
//             = H_{n-2}(z) - ((n-1)/z)*H_{n-1}(z) - [(-n/z^2)*H_n + (n/z)*dH_n/dz]
//             = H_{n-2}(z) - ((n-1)/z)*H_{n-1}(z) + (n/z^2)*H_n - (n/z)*[H_{n-1} - (n/z)*H_n]
//             = H_{n-2}(z) - ((n-1)/z)*H_{n-1}(z) + (n/z^2)*H_n - (n/z)*H_{n-1} + (n^2/z^2)*H_n
//             = H_{n-2}(z) - ((2n-1)/z)*H_{n-1}(z) + ((n+n^2)/z^2)*H_n
template <typename T>
T struve_h_zz_derivative(T n, T z) {
    const T eps = struve_h_eps<T>();

    if (std::abs(z) < eps) {
        // At z=0, second derivative is 0 for most n
        return T(0);
    }

    T h_nm2 = struve_h(n - T(2), z);
    T h_nm1 = struve_h(n - T(1), z);
    T h_n = struve_h(n, z);

    T z_inv = T(1) / z;
    T z_inv2 = z_inv * z_inv;

    return h_nm2 - (T(2)*n - T(1)) * z_inv * h_nm1 + (n + n*n) * z_inv2 * h_n;
}

// Mixed derivative d^2 H_n / dn dz (numerical)
template <typename T>
T struve_h_nz_derivative(T n, T z) {
    const T eps = std::sqrt(struve_h_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    // d/dn of dH_n/dz at n+h and n-h
    auto [grad_n_plus, grad_z_plus] = struve_h_backward(T(1), n + h, z);
    auto [grad_n_minus, grad_z_minus] = struve_h_backward(T(1), n - h, z);

    return (grad_z_plus - grad_z_minus) / (T(2) * h);
}

// Second derivative d^2 H_n / dn^2 (numerical)
template <typename T>
T struve_h_nn_derivative(T n, T z) {
    const T eps = std::cbrt(struve_h_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T h_plus = struve_h(n + h, z);
    T h_center = struve_h(n, z);
    T h_minus = struve_h(n - h, z);

    return (h_plus - T(2) * h_center + h_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> struve_h_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> four(T(4), T(0));

    c10::complex<T> h_nm2 = struve_h(n - two, z);
    c10::complex<T> h_n = struve_h(n, z);
    c10::complex<T> h_np2 = struve_h(n + two, z);

    return (h_nm2 - two * h_n + h_np2) / four;
}

template <typename T>
c10::complex<T> struve_h_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(struve_h_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    auto [grad_n_plus, grad_z_plus] = struve_h_backward(one, n + h, z);
    auto [grad_n_minus, grad_z_minus] = struve_h_backward(one, n - h, z);

    return (grad_z_plus - grad_z_minus) / (two * h);
}

template <typename T>
c10::complex<T> struve_h_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(struve_h_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> h_plus = struve_h(n + h, z);
    c10::complex<T> h_center = struve_h(n, z);
    c10::complex<T> h_minus = struve_h(n - h, z);

    return (h_plus - two * h_center + h_minus) / (h * h);
}

} // namespace detail

// Second-order backward pass
// Returns (grad_grad_output, grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> struve_h_backward_backward(
    T gg_n,
    T gg_z,
    T grad_output,
    T n,
    T z
) {
    const T pi = static_cast<T>(M_PI);
    const T eps = detail::struve_h_eps<T>();

    // First derivatives using correct formula: dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
    T h_nm1 = struve_h(n - T(1), z);
    T h_n = struve_h(n, z);

    T dh_dz;
    if (std::abs(z) < eps) {
        if (n == T(0)) {
            dh_dz = T(2) / pi;
        } else {
            dh_dz = T(0);
        }
    } else {
        dh_dz = h_nm1 - (n / z) * h_n;
    }
    T dh_dn = detail::struve_h_n_derivative(n, z);

    // Second derivatives
    T d2h_dz2 = detail::struve_h_zz_derivative(n, z);
    T d2h_dn2 = detail::struve_h_nn_derivative(n, z);
    T d2h_dndz = detail::struve_h_nz_derivative(n, z);

    // grad_grad_output = gg_n * dH/dn + gg_z * dH/dz
    T grad_grad_output = gg_n * dh_dn + gg_z * dh_dz;

    // grad_n = grad_output * (gg_n * d^2H/dn^2 + gg_z * d^2H/dndz)
    T grad_n = grad_output * (gg_n * d2h_dn2 + gg_z * d2h_dndz);

    // grad_z = grad_output * (gg_n * d^2H/dndz + gg_z * d^2H/dz^2)
    T grad_z = grad_output * (gg_n * d2h_dndz + gg_z * d2h_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> struve_h_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> zero(T(0), T(0));
    const T pi = static_cast<T>(M_PI);
    const T eps = detail::struve_h_eps<T>();

    // First derivatives using correct formula: dH_n/dz = H_{n-1}(z) - (n/z) * H_n(z)
    c10::complex<T> h_nm1 = struve_h(n - one, z);
    c10::complex<T> h_n = struve_h(n, z);

    c10::complex<T> dh_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dh_dz = two / c10::complex<T>(pi, T(0));
        } else {
            dh_dz = zero;
        }
    } else {
        dh_dz = h_nm1 - (n / z) * h_n;
    }
    c10::complex<T> dh_dn = detail::struve_h_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2h_dz2 = detail::struve_h_zz_derivative(n, z);
    c10::complex<T> d2h_dn2 = detail::struve_h_nn_derivative(n, z);
    c10::complex<T> d2h_dndz = detail::struve_h_nz_derivative(n, z);

    // Accumulate using conjugates for Wirtinger derivatives
    c10::complex<T> grad_grad_output = gg_n * std::conj(dh_dn) + gg_z * std::conj(dh_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2h_dn2) + gg_z * std::conj(d2h_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2h_dndz) + gg_z * std::conj(d2h_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
