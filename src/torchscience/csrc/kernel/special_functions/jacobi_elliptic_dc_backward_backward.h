#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "jacobi_elliptic_dc.h"
#include "jacobi_elliptic_dc_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Jacobi elliptic function dc(u, m)
//
// Uses numerical differentiation for computing second derivatives.
// This computes gradients for (grad_output, u, m) given the gradient-gradients
// (gg_u, gg_m) from the first backward pass.

namespace detail {

template <typename T>
T jacobi_elliptic_dc_d2u(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));

    T df_du_plus = jacobi_elliptic_dc_du(u + h, m);
    T df_du_minus = jacobi_elliptic_dc_du(u - h, m);

    return (df_du_plus - df_du_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_d2u(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> df_du_plus = jacobi_elliptic_dc_du(u + ch, m);
    c10::complex<T> df_du_minus = jacobi_elliptic_dc_du(u - ch, m);

    return (df_du_plus - df_du_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

template <typename T>
T jacobi_elliptic_dc_du_dm(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));

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

    T df_du_plus = jacobi_elliptic_dc_du(u, m_plus);
    T df_du_minus = jacobi_elliptic_dc_du(u, m_minus);

    return (df_du_plus - df_du_minus) / h_actual;
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_du_dm(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> df_du_plus = jacobi_elliptic_dc_du(u, m + ch);
    c10::complex<T> df_du_minus = jacobi_elliptic_dc_du(u, m - ch);

    return (df_du_plus - df_du_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

template <typename T>
T jacobi_elliptic_dc_d2m(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));

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

    T df_dm_plus = jacobi_elliptic_dc_dm(u, m_plus);
    T df_dm_minus = jacobi_elliptic_dc_dm(u, m_minus);

    return (df_dm_plus - df_dm_minus) / h_actual;
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_d2m(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(m)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> df_dm_plus = jacobi_elliptic_dc_dm(u, m + ch);
    c10::complex<T> df_dm_minus = jacobi_elliptic_dc_dm(u, m - ch);

    return (df_dm_plus - df_dm_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

template <typename T>
T jacobi_elliptic_dc_dm_du(T u, T m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));

    T df_dm_plus = jacobi_elliptic_dc_dm(u + h, m);
    T df_dm_minus = jacobi_elliptic_dc_dm(u - h, m);

    return (df_dm_plus - df_dm_minus) / (T(2) * h);
}

template <typename T>
c10::complex<T> jacobi_elliptic_dc_dm_du(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::sqrt(std::numeric_limits<T>::epsilon()) * std::max(static_cast<T>(std::abs(u)), T(1));
    c10::complex<T> ch(h, T(0));

    c10::complex<T> df_dm_plus = jacobi_elliptic_dc_dm(u + ch, m);
    c10::complex<T> df_dm_minus = jacobi_elliptic_dc_dm(u - ch, m);

    return (df_dm_plus - df_dm_minus) / (c10::complex<T>(T(2), T(0)) * ch);
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> jacobi_elliptic_dc_backward_backward(
    T gg_u,
    T gg_m,
    T gradient,
    T u,
    T m
) {
    // Get first derivatives
    T ddc_du = detail::jacobi_elliptic_dc_du(u, m);
    T ddc_dm = detail::jacobi_elliptic_dc_dm(u, m);

    // Get second derivatives
    T d2dc_du2 = detail::jacobi_elliptic_dc_d2u(u, m);
    T d2dc_du_dm = detail::jacobi_elliptic_dc_du_dm(u, m);
    T d2dc_dm2 = detail::jacobi_elliptic_dc_d2m(u, m);
    T d2dc_dm_du = detail::jacobi_elliptic_dc_dm_du(u, m);

    // Gradient w.r.t. grad_output
    T grad_grad_output = gg_u * ddc_du + gg_m * ddc_dm;

    // Gradient w.r.t. u
    T grad_u = gg_u * gradient * d2dc_du2 + gg_m * gradient * d2dc_dm_du;

    // Gradient w.r.t. m
    T grad_m = gg_u * gradient * d2dc_du_dm + gg_m * gradient * d2dc_dm2;

    return {grad_grad_output, grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_elliptic_dc_backward_backward(
    c10::complex<T> gg_u,
    c10::complex<T> gg_m,
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    // Get first derivatives
    c10::complex<T> ddc_du = detail::jacobi_elliptic_dc_du(u, m);
    c10::complex<T> ddc_dm = detail::jacobi_elliptic_dc_dm(u, m);

    // Get second derivatives
    c10::complex<T> d2dc_du2 = detail::jacobi_elliptic_dc_d2u(u, m);
    c10::complex<T> d2dc_du_dm = detail::jacobi_elliptic_dc_du_dm(u, m);
    c10::complex<T> d2dc_dm2 = detail::jacobi_elliptic_dc_d2m(u, m);
    c10::complex<T> d2dc_dm_du = detail::jacobi_elliptic_dc_dm_du(u, m);

    // For complex inputs with Wirtinger derivatives
    c10::complex<T> grad_grad_output = gg_u * std::conj(ddc_du) + gg_m * std::conj(ddc_dm);

    c10::complex<T> grad_u = gg_u * gradient * std::conj(d2dc_du2) + gg_m * gradient * std::conj(d2dc_dm_du);

    c10::complex<T> grad_m = gg_u * gradient * std::conj(d2dc_du_dm) + gg_m * gradient * std::conj(d2dc_dm2);

    return {grad_grad_output, grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
