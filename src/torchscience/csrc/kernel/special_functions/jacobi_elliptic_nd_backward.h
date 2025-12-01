#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_nd.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function nd(u, m)
//
// The gradients are computed using numerical differentiation for
// accuracy and stability. Since nd = 1/dn, we have:
// d(nd)/du = -1/dn^2 * d(dn)/du
// d(nd)/dm = -1/dn^2 * d(dn)/dm
//
// Using numerical differentiation for simplicity.

namespace detail {

// Compute d(nd)/du using finite differences
template <typename T>
T compute_dnd_du_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    T nd_plus = jacobi_elliptic_nd(u + h, m);
    T nd_minus = jacobi_elliptic_nd(u - h, m);

    return (nd_plus - nd_minus) / (T(2) * h);
}

// Compute d(nd)/dm using finite differences
template <typename T>
T compute_dnd_dm_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases for m near 0 or 1
    if (m < h) {
        // Forward difference near m = 0
        T nd_0 = jacobi_elliptic_nd(u, m);
        T nd_1 = jacobi_elliptic_nd(u, m + h);
        return (nd_1 - nd_0) / h;
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T nd_0 = jacobi_elliptic_nd(u, m);
        T nd_1 = jacobi_elliptic_nd(u, m - h);
        return (nd_0 - nd_1) / h;
    }

    // Central difference
    T nd_plus = jacobi_elliptic_nd(u, m + h);
    T nd_minus = jacobi_elliptic_nd(u, m - h);

    return (nd_plus - nd_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> compute_dnd_du_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nd_plus = jacobi_elliptic_nd(u + h_c, m);
    c10::complex<T> nd_minus = jacobi_elliptic_nd(u - h_c, m);

    return (nd_plus - nd_minus) / (two * h_c);
}

template <typename T>
c10::complex<T> compute_dnd_dm_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nd_plus = jacobi_elliptic_nd(u, m + h_c);
    c10::complex<T> nd_minus = jacobi_elliptic_nd(u, m - h_c);

    return (nd_plus - nd_minus) / (two * h_c);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_nd_backward(T gradient, T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    // nd(u, 0) = 1 for all u, so all gradients are 0
    if (std::abs(m) < eps) {
        return {T(0), T(0)};
    }

    // General case: use numerical differentiation for accuracy
    T dnd_du = detail::compute_dnd_du_numerical(u, m);
    T dnd_dm = detail::compute_dnd_dm_numerical(u, m);

    T grad_u = gradient * dnd_du;
    T grad_m = gradient * dnd_dm;

    return {grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_nd_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> zero(T(0), T(0));

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return {zero, zero};
    }

    // General case: use numerical differentiation
    c10::complex<T> dnd_du = detail::compute_dnd_du_numerical(u, m);
    c10::complex<T> dnd_dm = detail::compute_dnd_dm_numerical(u, m);

    // For complex inputs, use Wirtinger derivatives
    c10::complex<T> grad_u = gradient * std::conj(dnd_du);
    c10::complex<T> grad_m = gradient * std::conj(dnd_dm);

    return {grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
