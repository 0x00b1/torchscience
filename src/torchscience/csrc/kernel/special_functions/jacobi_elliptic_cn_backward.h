#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>
#include <vector>

#include "jacobi_elliptic_cn.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi elliptic function cn(u, m)
//
// Gradients:
// ∂cn/∂u = -sn(u, m) * dn(u, m)
//
// where:
// - sn(u, m) = sin(am(u, m))
// - dn(u, m) = sqrt(1 - m * sn^2(u, m))
//
// For ∂cn/∂m, we use numerical differentiation since the analytical
// formula involves partial derivatives of the amplitude function.

namespace detail {

// Helper to compute sn(u, m) and dn(u, m) using the AGM method
// Returns (sn, cn, dn) for efficiency
template <typename T>
std::tuple<T, T, T> jacobi_elliptic_all(T u, T m) {
    const T eps = jacobi_cn_eps<T>();

    // Special case: m = 0
    if (std::abs(m) < eps) {
        return {std::sin(u), std::cos(u), T(1)};
    }

    // Special case: m = 1
    if (std::abs(m - T(1)) < eps) {
        T sech_u = T(1) / std::cosh(u);
        return {std::tanh(u), sech_u, sech_u};
    }

    // Use the AGM method
    const int max_iter = jacobi_cn_max_iter<T>();
    std::vector<T> a_vals;
    std::vector<T> c_vals;
    a_vals.reserve(max_iter);
    c_vals.reserve(max_iter);

    T a = T(1);
    T b = std::sqrt(T(1) - m);
    T c = std::sqrt(m);

    a_vals.push_back(a);
    c_vals.push_back(c);

    int n = 0;
    while (std::abs(c) > eps && n < max_iter) {
        T a_new = (a + b) / T(2);
        T b_new = std::sqrt(a * b);
        c = (a - b) / T(2);

        a = a_new;
        b = b_new;

        a_vals.push_back(a);
        c_vals.push_back(c);
        ++n;
    }

    // Compute phi_n = 2^n * a_n * u
    T phi = std::ldexp(a * u, n);

    // Backward recurrence
    for (int i = n; i > 0; --i) {
        T sin_phi = std::sin(phi);
        T c_i = c_vals[i];
        T a_i = a_vals[i];
        T adjustment = std::asin(c_i / a_i * sin_phi);
        phi = (phi + adjustment) / T(2);
    }

    // am(u, m) = phi
    T sn = std::sin(phi);
    T cn = std::cos(phi);
    T dn = std::sqrt(T(1) - m * sn * sn);

    return {sn, cn, dn};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_elliptic_all(c10::complex<T> u, c10::complex<T> m) {
    const T eps = jacobi_cn_eps<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> two(T(2), T(0));

    // Special case: m ~ 0
    if (std::abs(m) < eps) {
        return {std::sin(u), std::cos(u), one};
    }

    // Special case: m ~ 1
    if (std::abs(m - one) < eps) {
        c10::complex<T> sech_u = one / std::cosh(u);
        return {std::tanh(u), sech_u, sech_u};
    }

    // Use the AGM method
    const int max_iter = jacobi_cn_max_iter<T>();
    std::vector<c10::complex<T>> a_vals;
    std::vector<c10::complex<T>> c_vals;
    a_vals.reserve(max_iter);
    c_vals.reserve(max_iter);

    c10::complex<T> a = one;
    c10::complex<T> b = std::sqrt(one - m);
    c10::complex<T> c = std::sqrt(m);

    a_vals.push_back(a);
    c_vals.push_back(c);

    int n = 0;
    while (std::abs(c) > eps && n < max_iter) {
        c10::complex<T> a_new = (a + b) / two;
        c10::complex<T> b_new = std::sqrt(a * b);
        c = (a - b) / two;

        a = a_new;
        b = b_new;

        a_vals.push_back(a);
        c_vals.push_back(c);
        ++n;
    }

    // Compute phi_n = 2^n * a_n * u
    c10::complex<T> scale(std::ldexp(T(1), n), T(0));
    c10::complex<T> phi = scale * a * u;

    // Backward recurrence
    for (int i = n; i > 0; --i) {
        c10::complex<T> sin_phi = std::sin(phi);
        c10::complex<T> c_i = c_vals[i];
        c10::complex<T> a_i = a_vals[i];
        c10::complex<T> adjustment = std::asin(c_i / a_i * sin_phi);
        phi = (phi + adjustment) / two;
    }

    // am(u, m) = phi
    c10::complex<T> sn = std::sin(phi);
    c10::complex<T> cn = std::cos(phi);
    c10::complex<T> dn = std::sqrt(one - m * sn * sn);

    return {sn, cn, dn};
}

// Numerical derivative of cn with respect to m using 5-point stencil
// for higher accuracy
template <typename T>
T jacobi_elliptic_cn_dm(T u, T m) {
    // Use a relative step size that balances truncation and round-off errors
    // For numerical differentiation, h ~ eps^(1/3) is optimal for central difference
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(m)), T(0.1));

    // Clamp h to stay within valid domain [0, 1]
    if (m < T(0.5)) {
        h = std::min(h, m / T(2));  // Don't go below 0
        h = std::min(h, (T(1) - m) / T(2));  // Don't go above 1
    } else {
        h = std::min(h, (T(1) - m) / T(2));  // Don't go above 1
        h = std::min(h, m / T(2));  // Don't go below 0
    }

    // Ensure minimum step size
    h = std::max(h, T(1e-8));

    // Use 5-point stencil for higher accuracy:
    // f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    T f_p2h = jacobi_elliptic_cn(u, m + T(2) * h);
    T f_ph = jacobi_elliptic_cn(u, m + h);
    T f_mh = jacobi_elliptic_cn(u, m - h);
    T f_m2h = jacobi_elliptic_cn(u, m - T(2) * h);

    return (-f_p2h + T(8) * f_ph - T(8) * f_mh + f_m2h) / (T(12) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn_dm(c10::complex<T> u, c10::complex<T> m) {
    // Use 5-point stencil in the complex plane for better accuracy
    const T h_rel = std::cbrt(std::numeric_limits<T>::epsilon());
    T h = h_rel * std::max(static_cast<T>(std::abs(m)), T(0.1));
    h = std::max(h, T(1e-8));
    c10::complex<T> ch(h, T(0));

    // 5-point stencil: f'(x) = (-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)) / (12h)
    c10::complex<T> f_p2h = jacobi_elliptic_cn(u, m + c10::complex<T>(T(2), T(0)) * ch);
    c10::complex<T> f_ph = jacobi_elliptic_cn(u, m + ch);
    c10::complex<T> f_mh = jacobi_elliptic_cn(u, m - ch);
    c10::complex<T> f_m2h = jacobi_elliptic_cn(u, m - c10::complex<T>(T(2), T(0)) * ch);

    c10::complex<T> eight(T(8), T(0));
    c10::complex<T> twelve(T(12), T(0));

    return (-f_p2h + eight * f_ph - eight * f_mh + f_m2h) / (twelve * ch);
}

} // namespace detail

template <typename T>
std::tuple<T, T> jacobi_elliptic_cn_backward(T gradient, T u, T m) {
    // Get sn, cn, dn
    auto [sn, cn, dn] = detail::jacobi_elliptic_all(u, m);

    // ∂cn/∂u = -sn * dn
    T dcn_du = -sn * dn;

    // ∂cn/∂m using numerical differentiation
    T dcn_dm = detail::jacobi_elliptic_cn_dm(u, m);

    return {gradient * dcn_du, gradient * dcn_dm};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
jacobi_elliptic_cn_backward(c10::complex<T> gradient, c10::complex<T> u, c10::complex<T> m) {
    // Get sn, cn, dn
    auto [sn, cn, dn] = detail::jacobi_elliptic_all(u, m);

    // ∂cn/∂u = -sn * dn
    c10::complex<T> dcn_du = -sn * dn;

    // ∂cn/∂m using numerical differentiation
    c10::complex<T> dcn_dm = detail::jacobi_elliptic_cn_dm(u, m);

    // For complex inputs with Wirtinger derivatives
    return {gradient * std::conj(dcn_du), gradient * std::conj(dcn_dm)};
}

} // namespace torchscience::kernel::special_functions
