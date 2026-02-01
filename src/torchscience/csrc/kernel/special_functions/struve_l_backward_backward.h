#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "struve_l.h"
#include "struve_l_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative d^2 L_n / dz^2
// Using dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)  (same as H_n)
// d^2L_n/dz^2 = dL_{n-1}/dz - d/dz[(n/z) * L_n(z)]
//             = L_{n-2}(z) - ((n-1)/z)*L_{n-1}(z) - [(-n/z^2)*L_n + (n/z)*dL_n/dz]
//             = L_{n-2}(z) - ((n-1)/z)*L_{n-1}(z) + (n/z^2)*L_n - (n/z)*[L_{n-1} - (n/z)*L_n]
//             = L_{n-2}(z) - ((n-1)/z)*L_{n-1}(z) + (n/z^2)*L_n - (n/z)*L_{n-1} + (n^2/z^2)*L_n
//             = L_{n-2}(z) - ((2n-1)/z)*L_{n-1}(z) + ((n+n^2)/z^2)*L_n
template <typename T>
T struve_l_zz_derivative(T n, T z) {
    const T eps = struve_l_eps<T>();

    if (std::abs(z) < eps) {
        // At z=0, second derivative is 0 for most n
        return T(0);
    }

    T l_nm2 = struve_l(n - T(2), z);
    T l_nm1 = struve_l(n - T(1), z);
    T l_n = struve_l(n, z);

    T z_inv = T(1) / z;
    T z_inv2 = z_inv * z_inv;

    // Same signs as H_n (MINUS sign for L_{n-1} term)
    return l_nm2 - (T(2)*n - T(1)) * z_inv * l_nm1 + (n + n*n) * z_inv2 * l_n;
}

// Mixed derivative d^2 L_n / dn dz (numerical)
template <typename T>
T struve_l_nz_derivative(T n, T z) {
    const T eps = std::sqrt(struve_l_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    auto [grad_n_plus, grad_z_plus] = struve_l_backward(T(1), n + h, z);
    auto [grad_n_minus, grad_z_minus] = struve_l_backward(T(1), n - h, z);

    return (grad_z_plus - grad_z_minus) / (T(2) * h);
}

// Second derivative d^2 L_n / dn^2 (numerical)
template <typename T>
T struve_l_nn_derivative(T n, T z) {
    const T eps = std::cbrt(struve_l_eps<T>());
    T h = eps * (std::abs(n) > T(1) ? std::abs(n) : T(1));

    T l_plus = struve_l(n + h, z);
    T l_center = struve_l(n, z);
    T l_minus = struve_l(n - h, z);

    return (l_plus - T(2) * l_center + l_minus) / (h * h);
}

// Complex versions
template <typename T>
c10::complex<T> struve_l_zz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const T eps = struve_l_eps<T>();

    if (std::abs(z) < eps) {
        return c10::complex<T>(T(0), T(0));
    }

    c10::complex<T> l_nm2 = struve_l(n - two, z);
    c10::complex<T> l_nm1 = struve_l(n - one, z);
    c10::complex<T> l_n = struve_l(n, z);

    c10::complex<T> z_inv = one / z;
    c10::complex<T> z_inv2 = z_inv * z_inv;

    // Same signs as H_n (MINUS sign for L_{n-1} term)
    return l_nm2 - (two*n - one) * z_inv * l_nm1 + (n + n*n) * z_inv2 * l_n;
}

template <typename T>
c10::complex<T> struve_l_nz_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::sqrt(struve_l_eps<T>());
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    auto [grad_n_plus, grad_z_plus] = struve_l_backward(one, n + h, z);
    auto [grad_n_minus, grad_z_minus] = struve_l_backward(one, n - h, z);

    return (grad_z_plus - grad_z_minus) / (two * h);
}

template <typename T>
c10::complex<T> struve_l_nn_derivative(c10::complex<T> n, c10::complex<T> z) {
    const T eps = std::cbrt(struve_l_eps<T>());
    const c10::complex<T> two(T(2), T(0));

    T n_mag = std::abs(n);
    c10::complex<T> h(eps * (n_mag > T(1) ? n_mag : T(1)), T(0));

    c10::complex<T> l_plus = struve_l(n + h, z);
    c10::complex<T> l_center = struve_l(n, z);
    c10::complex<T> l_minus = struve_l(n - h, z);

    return (l_plus - two * l_center + l_minus) / (h * h);
}

} // namespace detail

// Second-order backward pass
// Returns (grad_grad_output, grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> struve_l_backward_backward(
    T gg_n,
    T gg_z,
    T grad_output,
    T n,
    T z
) {
    const T pi = static_cast<T>(M_PI);
    const T eps = detail::struve_l_eps<T>();

    // First derivatives using correct formula: dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)
    T l_nm1 = struve_l(n - T(1), z);
    T l_n = struve_l(n, z);

    T dl_dz;
    if (std::abs(z) < eps) {
        if (n == T(0)) {
            dl_dz = T(2) / pi;
        } else {
            dl_dz = T(0);
        }
    } else {
        // MINUS sign for (n/z)*L_n (same as H_n)
        dl_dz = l_nm1 - (n / z) * l_n;
    }
    T dl_dn = detail::struve_l_n_derivative(n, z);

    // Second derivatives
    T d2l_dz2 = detail::struve_l_zz_derivative(n, z);
    T d2l_dn2 = detail::struve_l_nn_derivative(n, z);
    T d2l_dndz = detail::struve_l_nz_derivative(n, z);

    // grad_grad_output = gg_n * dL/dn + gg_z * dL/dz
    T grad_grad_output = gg_n * dl_dn + gg_z * dl_dz;

    // grad_n = grad_output * (gg_n * d^2L/dn^2 + gg_z * d^2L/dndz)
    T grad_n = grad_output * (gg_n * d2l_dn2 + gg_z * d2l_dndz);

    // grad_z = grad_output * (gg_n * d^2L/dndz + gg_z * d^2L/dz^2)
    T grad_z = grad_output * (gg_n * d2l_dndz + gg_z * d2l_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> struve_l_backward_backward(
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
    const T eps = detail::struve_l_eps<T>();

    // First derivatives using correct formula: dL_n/dz = L_{n-1}(z) - (n/z) * L_n(z)
    c10::complex<T> l_nm1 = struve_l(n - one, z);
    c10::complex<T> l_n = struve_l(n, z);

    c10::complex<T> dl_dz;
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            dl_dz = two / c10::complex<T>(pi, T(0));
        } else {
            dl_dz = zero;
        }
    } else {
        // MINUS sign for (n/z)*L_n (same as H_n)
        dl_dz = l_nm1 - (n / z) * l_n;
    }
    c10::complex<T> dl_dn = detail::struve_l_n_derivative(n, z);

    // Second derivatives
    c10::complex<T> d2l_dz2 = detail::struve_l_zz_derivative(n, z);
    c10::complex<T> d2l_dn2 = detail::struve_l_nn_derivative(n, z);
    c10::complex<T> d2l_dndz = detail::struve_l_nz_derivative(n, z);

    // Accumulate using conjugates for Wirtinger derivatives
    c10::complex<T> grad_grad_output = gg_n * std::conj(dl_dn) + gg_z * std::conj(dl_dz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2l_dn2) + gg_z * std::conj(d2l_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2l_dndz) + gg_z * std::conj(d2l_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
