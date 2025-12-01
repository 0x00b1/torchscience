#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_ns.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function ns(u, m)
//
// The gradients are computed using numerical differentiation for
// accuracy and stability. Since ns = 1/sn, we have:
// d(ns)/du = -1/sn^2 * d(sn)/du
// d(ns)/dm = -1/sn^2 * d(sn)/dm
//
// Using numerical differentiation for simplicity.

namespace detail {

// Compute d(ns)/du using finite differences
template <typename T>
T compute_dns_du_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    T ns_plus = jacobi_elliptic_ns(u + h, m);
    T ns_minus = jacobi_elliptic_ns(u - h, m);

    return (ns_plus - ns_minus) / (T(2) * h);
}

// Compute d(ns)/dm using finite differences
template <typename T>
T compute_dns_dm_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases for m near 0 or 1
    if (m < h) {
        // Forward difference near m = 0
        T ns_0 = jacobi_elliptic_ns(u, m);
        T ns_1 = jacobi_elliptic_ns(u, m + h);
        return (ns_1 - ns_0) / h;
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T ns_0 = jacobi_elliptic_ns(u, m);
        T ns_1 = jacobi_elliptic_ns(u, m - h);
        return (ns_0 - ns_1) / h;
    }

    // Central difference
    T ns_plus = jacobi_elliptic_ns(u, m + h);
    T ns_minus = jacobi_elliptic_ns(u, m - h);

    return (ns_plus - ns_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> compute_dns_du_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> ns_plus = jacobi_elliptic_ns(u + h_c, m);
    c10::complex<T> ns_minus = jacobi_elliptic_ns(u - h_c, m);

    return (ns_plus - ns_minus) / (two * h_c);
}

template <typename T>
c10::complex<T> compute_dns_dm_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> ns_plus = jacobi_elliptic_ns(u, m + h_c);
    c10::complex<T> ns_minus = jacobi_elliptic_ns(u, m - h_c);

    return (ns_plus - ns_minus) / (two * h_c);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_ns_backward(T gradient, T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    // ns(u, 0) = csc(u) = 1/sin(u)
    // d(ns)/du = -csc(u)*cot(u) = -cos(u)/sin^2(u)
    // d(ns)/dm uses numerical differentiation
    if (std::abs(m) < eps) {
        T cos_u = std::cos(u);
        T sin_u = std::sin(u);
        T grad_u = gradient * (-cos_u / (sin_u * sin_u));
        T grad_m = gradient * detail::compute_dns_dm_numerical(u, m);
        return {grad_u, grad_m};
    }

    // General case: use numerical differentiation for accuracy
    T dns_du = detail::compute_dns_du_numerical(u, m);
    T dns_dm = detail::compute_dns_dm_numerical(u, m);

    T grad_u = gradient * dns_du;
    T grad_m = gradient * dns_dm;

    return {grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_ns_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> zero(T(0), T(0));

    // General case: use numerical differentiation
    c10::complex<T> dns_du = detail::compute_dns_du_numerical(u, m);
    c10::complex<T> dns_dm = detail::compute_dns_dm_numerical(u, m);

    // For complex inputs, use Wirtinger derivatives
    c10::complex<T> grad_u = gradient * std::conj(dns_du);
    c10::complex<T> grad_m = gradient * std::conj(dns_dm);

    return {grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
