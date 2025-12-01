#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_dn.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function dn(u, m)
//
// The gradients are computed using numerical differentiation for
// accuracy and stability. The analytical formulas involve partial
// derivatives of the amplitude function that are complex to derive
// correctly.

namespace detail {

// Compute sn(u, m) = sin(am(u, m))
template <typename T>
T jacobi_elliptic_sn(T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return std::sin(u);
    }

    // Special case: m = 1
    if (std::abs(m - T(1)) < eps) {
        return std::tanh(u);
    }

    T am = jacobi_amplitude_am(u, m);
    return std::sin(am);
}

// Compute cn(u, m) = cos(am(u, m))
template <typename T>
T jacobi_elliptic_cn(T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return std::cos(u);
    }

    // Special case: m = 1
    if (std::abs(m - T(1)) < eps) {
        return T(1) / std::cosh(u);
    }

    T am = jacobi_amplitude_am(u, m);
    return std::cos(am);
}

template <typename T>
c10::complex<T> jacobi_elliptic_sn(c10::complex<T> u, c10::complex<T> m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> one(T(1), T(0));

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return std::sin(u);
    }

    // Special case: m = 1
    if (std::abs(m - one) < eps) {
        return std::tanh(u);
    }

    c10::complex<T> am = jacobi_amplitude_am(u, m);
    return std::sin(am);
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn(c10::complex<T> u, c10::complex<T> m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> one(T(1), T(0));

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return std::cos(u);
    }

    // Special case: m = 1
    if (std::abs(m - one) < eps) {
        return one / std::cosh(u);
    }

    c10::complex<T> am = jacobi_amplitude_am(u, m);
    return std::cos(am);
}

// Compute d(dn)/du using finite differences
template <typename T>
T compute_ddn_du_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    T dn_plus = jacobi_elliptic_dn(u + h, m);
    T dn_minus = jacobi_elliptic_dn(u - h, m);

    return (dn_plus - dn_minus) / (T(2) * h);
}

// Compute d(dn)/dm using finite differences
template <typename T>
T compute_ddn_dm_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases for m near 0 or 1
    if (m < h) {
        // Forward difference near m = 0
        T dn_0 = jacobi_elliptic_dn(u, m);
        T dn_1 = jacobi_elliptic_dn(u, m + h);
        return (dn_1 - dn_0) / h;
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T dn_0 = jacobi_elliptic_dn(u, m);
        T dn_1 = jacobi_elliptic_dn(u, m - h);
        return (dn_0 - dn_1) / h;
    }

    // Central difference
    T dn_plus = jacobi_elliptic_dn(u, m + h);
    T dn_minus = jacobi_elliptic_dn(u, m - h);

    return (dn_plus - dn_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> compute_ddn_du_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> dn_plus = jacobi_elliptic_dn(u + h_c, m);
    c10::complex<T> dn_minus = jacobi_elliptic_dn(u - h_c, m);

    return (dn_plus - dn_minus) / (two * h_c);
}

template <typename T>
c10::complex<T> compute_ddn_dm_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> dn_plus = jacobi_elliptic_dn(u, m + h_c);
    c10::complex<T> dn_minus = jacobi_elliptic_dn(u, m - h_c);

    return (dn_plus - dn_minus) / (two * h_c);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_dn_backward(T gradient, T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    // dn(u, 0) = 1 for all u, so all gradients are 0
    if (std::abs(m) < eps) {
        return {T(0), T(0)};
    }

    // Special case: m = 1
    // dn(u, 1) = sech(u), so d(dn)/du = -sech(u)*tanh(u)
    if (std::abs(m - T(1)) < eps) {
        T sech_u = T(1) / std::cosh(u);
        T tanh_u = std::tanh(u);
        T grad_u = gradient * (-sech_u * tanh_u);

        // Use numerical differentiation for d(dn)/dm
        T ddn_dm = detail::compute_ddn_dm_numerical(u, m);
        T grad_m = gradient * ddn_dm;

        return {grad_u, grad_m};
    }

    // General case: use numerical differentiation for accuracy
    T ddn_du = detail::compute_ddn_du_numerical(u, m);
    T ddn_dm = detail::compute_ddn_dm_numerical(u, m);

    T grad_u = gradient * ddn_du;
    T grad_m = gradient * ddn_dm;

    return {grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_dn_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return {zero, zero};
    }

    // Special case: m = 1
    if (std::abs(m - one) < eps) {
        c10::complex<T> sech_u = one / std::cosh(u);
        c10::complex<T> tanh_u = std::tanh(u);
        c10::complex<T> ddn_du = -sech_u * tanh_u;
        c10::complex<T> grad_u = gradient * std::conj(ddn_du);

        // Use numerical differentiation for d(dn)/dm
        c10::complex<T> ddn_dm = detail::compute_ddn_dm_numerical(u, m);
        c10::complex<T> grad_m = gradient * std::conj(ddn_dm);

        return {grad_u, grad_m};
    }

    // General case: use numerical differentiation
    c10::complex<T> ddn_du = detail::compute_ddn_du_numerical(u, m);
    c10::complex<T> ddn_dm = detail::compute_ddn_dm_numerical(u, m);

    // For complex inputs, use Wirtinger derivatives
    c10::complex<T> grad_u = gradient * std::conj(ddn_du);
    c10::complex<T> grad_m = gradient * std::conj(ddn_dm);

    return {grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
