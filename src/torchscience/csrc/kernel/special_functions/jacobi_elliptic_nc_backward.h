#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_nc.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function nc(u, m)
//
// The gradients are computed using numerical differentiation for
// accuracy and stability. Since nc = 1/cn, we have:
// d(nc)/du = -1/cn^2 * d(cn)/du
// d(nc)/dm = -1/cn^2 * d(cn)/dm
//
// Using numerical differentiation for simplicity.

namespace detail {

// Compute d(nc)/du using finite differences
template <typename T>
T compute_dnc_du_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    T nc_plus = jacobi_elliptic_nc(u + h, m);
    T nc_minus = jacobi_elliptic_nc(u - h, m);

    return (nc_plus - nc_minus) / (T(2) * h);
}

// Compute d(nc)/dm using finite differences
template <typename T>
T compute_dnc_dm_numerical(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases for m near 0 or 1
    if (m < h) {
        // Forward difference near m = 0
        T nc_0 = jacobi_elliptic_nc(u, m);
        T nc_1 = jacobi_elliptic_nc(u, m + h);
        return (nc_1 - nc_0) / h;
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T nc_0 = jacobi_elliptic_nc(u, m);
        T nc_1 = jacobi_elliptic_nc(u, m - h);
        return (nc_0 - nc_1) / h;
    }

    // Central difference
    T nc_plus = jacobi_elliptic_nc(u, m + h);
    T nc_minus = jacobi_elliptic_nc(u, m - h);

    return (nc_plus - nc_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> compute_dnc_du_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nc_plus = jacobi_elliptic_nc(u + h_c, m);
    c10::complex<T> nc_minus = jacobi_elliptic_nc(u - h_c, m);

    return (nc_plus - nc_minus) / (two * h_c);
}

template <typename T>
c10::complex<T> compute_dnc_dm_numerical(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nc_plus = jacobi_elliptic_nc(u, m + h_c);
    c10::complex<T> nc_minus = jacobi_elliptic_nc(u, m - h_c);

    return (nc_plus - nc_minus) / (two * h_c);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_nc_backward(T gradient, T u, T m) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Special case: m = 0
    // nc(u, 0) = sec(u) = 1/cos(u)
    // d(nc)/du = sec(u)*tan(u) = sin(u)/cos^2(u)
    // d(nc)/dm uses numerical differentiation
    if (std::abs(m) < eps) {
        T cos_u = std::cos(u);
        T sin_u = std::sin(u);
        T grad_u = gradient * sin_u / (cos_u * cos_u);
        T grad_m = gradient * detail::compute_dnc_dm_numerical(u, m);
        return {grad_u, grad_m};
    }

    // General case: use numerical differentiation for accuracy
    T dnc_du = detail::compute_dnc_du_numerical(u, m);
    T dnc_dm = detail::compute_dnc_dm_numerical(u, m);

    T grad_u = gradient * dnc_du;
    T grad_m = gradient * dnc_dm;

    return {grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> jacobi_elliptic_nc_backward(
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    const T eps = std::numeric_limits<T>::epsilon() * T(10);
    c10::complex<T> zero(T(0), T(0));

    // General case: use numerical differentiation
    c10::complex<T> dnc_du = detail::compute_dnc_du_numerical(u, m);
    c10::complex<T> dnc_dm = detail::compute_dnc_dm_numerical(u, m);

    // For complex inputs, use Wirtinger derivatives
    c10::complex<T> grad_u = gradient * std::conj(dnc_du);
    c10::complex<T> grad_m = gradient * std::conj(dnc_dm);

    return {grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
