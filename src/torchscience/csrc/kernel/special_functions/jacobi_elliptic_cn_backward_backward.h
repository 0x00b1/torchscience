#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_cn.h"
#include "jacobi_elliptic_cn_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Jacobi elliptic function cn(u, m)
//
// Uses numerical differentiation for computing second derivatives.
// This computes gradients for (grad_output, u, m) given the gradient-gradients
// (gg_u, gg_m) from the first backward pass.

namespace detail {

// Compute d(dcn/du)/du = d(-sn * dn)/du = -cn * dn^2 - m * sn^2 * cn / dn
// But we use numerical differentiation for robustness
template <typename T>
T jacobi_elliptic_cn_d2u(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));

    // Central difference for d(dcn/du)/du
    auto [sn_plus, cn_plus, dn_plus] = jacobi_elliptic_all(u + h, m);
    auto [sn_minus, cn_minus, dn_minus] = jacobi_elliptic_all(u - h, m);

    T dcn_du_plus = -sn_plus * dn_plus;
    T dcn_du_minus = -sn_minus * dn_minus;

    return (dcn_du_plus - dcn_du_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn_d2u(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));
    c10::complex<T> ch(h, T(0));

    auto [sn_plus, cn_plus, dn_plus] = jacobi_elliptic_all(u + ch, m);
    auto [sn_minus, cn_minus, dn_minus] = jacobi_elliptic_all(u - ch, m);

    c10::complex<T> dcn_du_plus = -sn_plus * dn_plus;
    c10::complex<T> dcn_du_minus = -sn_minus * dn_minus;

    return (dcn_du_plus - dcn_du_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

// Compute d(dcn/du)/dm using numerical differentiation
template <typename T>
T jacobi_elliptic_cn_du_dm(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));

    // Handle boundary cases
    T h_actual = h;
    T m_plus = m + h;
    T m_minus = m - h;

    if (m_minus < T(0)) {
        m_minus = T(0);
        h_actual = m_plus - m_minus;
    }
    if (m_plus > T(1)) {
        m_plus = T(1);
        h_actual = m_plus - m_minus;
    }

    auto [sn_plus, cn_plus, dn_plus] = jacobi_elliptic_all(u, m_plus);
    auto [sn_minus, cn_minus, dn_minus] = jacobi_elliptic_all(u, m_minus);

    T dcn_du_plus = -sn_plus * dn_plus;
    T dcn_du_minus = -sn_minus * dn_minus;

    return (dcn_du_plus - dcn_du_minus) / h_actual;
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn_du_dm(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));
    c10::complex<T> ch(h, T(0));

    auto [sn_plus, cn_plus, dn_plus] = jacobi_elliptic_all(u, m + ch);
    auto [sn_minus, cn_minus, dn_minus] = jacobi_elliptic_all(u, m - ch);

    c10::complex<T> dcn_du_plus = -sn_plus * dn_plus;
    c10::complex<T> dcn_du_minus = -sn_minus * dn_minus;

    return (dcn_du_plus - dcn_du_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

// Compute d(dcn/dm)/dm using numerical differentiation
template <typename T>
T jacobi_elliptic_cn_d2m(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));

    // Handle boundary cases
    T h_actual = h;
    T m_plus = m + h;
    T m_minus = m - h;

    if (m_minus < T(0)) {
        m_minus = T(0);
        h_actual = m_plus - m_minus;
    }
    if (m_plus > T(1)) {
        m_plus = T(1);
        h_actual = m_plus - m_minus;
    }

    T dcn_dm_plus = jacobi_elliptic_cn_dm(u, m_plus);
    T dcn_dm_minus = jacobi_elliptic_cn_dm(u, m_minus);

    return (dcn_dm_plus - dcn_dm_minus) / h_actual;
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn_d2m(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> dcn_dm_plus = jacobi_elliptic_cn_dm(u, m + ch);
    c10::complex<T> dcn_dm_minus = jacobi_elliptic_cn_dm(u, m - ch);

    return (dcn_dm_plus - dcn_dm_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

// Compute d(dcn/dm)/du using numerical differentiation
template <typename T>
T jacobi_elliptic_cn_dm_du(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));

    T dcn_dm_plus = jacobi_elliptic_cn_dm(u + h, m);
    T dcn_dm_minus = jacobi_elliptic_cn_dm(u - h, m);

    return (dcn_dm_plus - dcn_dm_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_cn_dm_du(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> dcn_dm_plus = jacobi_elliptic_cn_dm(u + ch, m);
    c10::complex<T> dcn_dm_minus = jacobi_elliptic_cn_dm(u - ch, m);

    return (dcn_dm_plus - dcn_dm_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> jacobi_elliptic_cn_backward_backward(
    T gg_u,
    T gg_m,
    T gradient,
    T u,
    T m
) {
    // Get first derivatives
    auto [sn, cn, dn] = detail::jacobi_elliptic_all(u, m);
    T dcn_du = -sn * dn;
    T dcn_dm = detail::jacobi_elliptic_cn_dm(u, m);

    // Get second derivatives
    T d2cn_du2 = detail::jacobi_elliptic_cn_d2u(u, m);
    T d2cn_du_dm = detail::jacobi_elliptic_cn_du_dm(u, m);
    T d2cn_dm2 = detail::jacobi_elliptic_cn_d2m(u, m);

    // Gradient w.r.t. grad_output:
    // The backward pass computes: grad_u = gradient * dcn_du, grad_m = gradient * dcn_dm
    // So: d(grad_u)/d(gradient) = dcn_du, d(grad_m)/d(gradient) = dcn_dm
    // Therefore: grad_grad_output = gg_u * dcn_du + gg_m * dcn_dm
    T grad_grad_output = gg_u * dcn_du + gg_m * dcn_dm;

    // Gradient w.r.t. u:
    // d(grad_u)/du = gradient * d2cn_du2
    // d(grad_m)/du = gradient * d(dcn_dm)/du
    T grad_u = gg_u * gradient * d2cn_du2 + gg_m * gradient * detail::jacobi_elliptic_cn_dm_du(u, m);

    // Gradient w.r.t. m:
    // d(grad_u)/dm = gradient * d(dcn_du)/dm
    // d(grad_m)/dm = gradient * d2cn_dm2
    T grad_m = gg_u * gradient * d2cn_du_dm + gg_m * gradient * d2cn_dm2;

    return {grad_grad_output, grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_elliptic_cn_backward_backward(
    c10::complex<T> gg_u,
    c10::complex<T> gg_m,
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    // Get first derivatives
    auto [sn, cn, dn] = detail::jacobi_elliptic_all(u, m);
    c10::complex<T> dcn_du = -sn * dn;
    c10::complex<T> dcn_dm = detail::jacobi_elliptic_cn_dm(u, m);

    // Get second derivatives
    c10::complex<T> d2cn_du2 = detail::jacobi_elliptic_cn_d2u(u, m);
    c10::complex<T> d2cn_du_dm = detail::jacobi_elliptic_cn_du_dm(u, m);
    c10::complex<T> d2cn_dm2 = detail::jacobi_elliptic_cn_d2m(u, m);
    c10::complex<T> d2cn_dm_du = detail::jacobi_elliptic_cn_dm_du(u, m);

    // For complex inputs with Wirtinger derivatives
    c10::complex<T> grad_grad_output = gg_u * std::conj(dcn_du) + gg_m * std::conj(dcn_dm);

    c10::complex<T> grad_u = gg_u * gradient * std::conj(d2cn_du2) + gg_m * gradient * std::conj(d2cn_dm_du);

    c10::complex<T> grad_m = gg_u * gradient * std::conj(d2cn_du_dm) + gg_m * gradient * std::conj(d2cn_dm2);

    return {grad_grad_output, grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
