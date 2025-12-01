#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_nd.h"
#include "jacobi_elliptic_nd_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Jacobi elliptic function nd(u, m)
//
// This computes the gradients of the backward pass outputs with respect to
// their inputs. Uses numerical differentiation for stability.
//
// Inputs:
//   gg_u: gradient w.r.t. grad_u from backward pass
//   gg_m: gradient w.r.t. grad_m from backward pass
//   grad: original gradient from forward pass
//   u: argument
//   m: parameter
//
// Outputs:
//   grad_grad: gradient w.r.t. grad
//   grad_u: gradient w.r.t. u
//   grad_m: gradient w.r.t. m

namespace detail {

// Compute partial derivatives using finite differences
template <typename T>
T compute_d2nd_du2(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    // Central difference: d^2(nd)/du^2 = (nd(u+h) - 2*nd(u) + nd(u-h)) / h^2
    T nd_plus = jacobi_elliptic_nd(u + h, m);
    T nd_center = jacobi_elliptic_nd(u, m);
    T nd_minus = jacobi_elliptic_nd(u - h, m);

    return (nd_plus - T(2) * nd_center + nd_minus) / (h * h);
}

template <typename T>
T compute_d2nd_dm2(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases
    if (m < h) {
        // Forward difference near m = 0
        T nd_0 = jacobi_elliptic_nd(u, m);
        T nd_1 = jacobi_elliptic_nd(u, m + h);
        T nd_2 = jacobi_elliptic_nd(u, m + T(2) * h);
        return (nd_2 - T(2) * nd_1 + nd_0) / (h * h);
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T nd_0 = jacobi_elliptic_nd(u, m);
        T nd_1 = jacobi_elliptic_nd(u, m - h);
        T nd_2 = jacobi_elliptic_nd(u, m - T(2) * h);
        return (nd_2 - T(2) * nd_1 + nd_0) / (h * h);
    }

    // Central difference
    T nd_plus = jacobi_elliptic_nd(u, m + h);
    T nd_center = jacobi_elliptic_nd(u, m);
    T nd_minus = jacobi_elliptic_nd(u, m - h);

    return (nd_plus - T(2) * nd_center + nd_minus) / (h * h);
}

template <typename T>
T compute_d2nd_dudm(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/4.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Mixed partial: d^2(nd)/du/dm
    // Use central differences: (nd(u+h,m+h) - nd(u+h,m-h) - nd(u-h,m+h) + nd(u-h,m-h)) / (4*h^2)

    T m_plus = m + h;
    T m_minus = m - h;

    // Handle boundary in m
    if (m < h) {
        m_minus = m;
        m_plus = m + T(2) * h;
    } else if (m > T(1) - h) {
        m_plus = m;
        m_minus = m - T(2) * h;
    }

    T f_pp = jacobi_elliptic_nd(u + h, m_plus);
    T f_pm = jacobi_elliptic_nd(u + h, m_minus);
    T f_mp = jacobi_elliptic_nd(u - h, m_plus);
    T f_mm = jacobi_elliptic_nd(u - h, m_minus);

    T h_m = m_plus - m_minus;
    return (f_pp - f_pm - f_mp + f_mm) / (T(4) * h * h_m / T(2));
}

template <typename T>
c10::complex<T> compute_d2nd_du2(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nd_plus = jacobi_elliptic_nd(u + h_c, m);
    c10::complex<T> nd_center = jacobi_elliptic_nd(u, m);
    c10::complex<T> nd_minus = jacobi_elliptic_nd(u - h_c, m);

    return (nd_plus - two * nd_center + nd_minus) / (h_c * h_c);
}

template <typename T>
c10::complex<T> compute_d2nd_dm2(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> nd_plus = jacobi_elliptic_nd(u, m + h_c);
    c10::complex<T> nd_center = jacobi_elliptic_nd(u, m);
    c10::complex<T> nd_minus = jacobi_elliptic_nd(u, m - h_c);

    return (nd_plus - two * nd_center + nd_minus) / (h_c * h_c);
}

template <typename T>
c10::complex<T> compute_d2nd_dudm(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/4.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> f_pp = jacobi_elliptic_nd(u + h_c, m + h_c);
    c10::complex<T> f_pm = jacobi_elliptic_nd(u + h_c, m - h_c);
    c10::complex<T> f_mp = jacobi_elliptic_nd(u - h_c, m + h_c);
    c10::complex<T> f_mm = jacobi_elliptic_nd(u - h_c, m - h_c);

    return (f_pp - f_pm - f_mp + f_mm) / (four * h_c * h_c);
}

// Compute dnd/du
template <typename T>
T compute_dnd_du(T u, T m) {
    auto [grad_u, grad_m] = jacobi_elliptic_nd_backward(T(1), u, m);
    return grad_u;
}

// Compute dnd/dm
template <typename T>
T compute_dnd_dm(T u, T m) {
    auto [grad_u, grad_m] = jacobi_elliptic_nd_backward(T(1), u, m);
    return grad_m;
}

template <typename T>
c10::complex<T> compute_dnd_du(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> one(T(1), T(0));
    auto [grad_u, grad_m] = jacobi_elliptic_nd_backward(one, u, m);
    return grad_u;
}

template <typename T>
c10::complex<T> compute_dnd_dm(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> one(T(1), T(0));
    auto [grad_u, grad_m] = jacobi_elliptic_nd_backward(one, u, m);
    return grad_m;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> jacobi_elliptic_nd_backward_backward(
    T gg_u,
    T gg_m,
    T grad,
    T u,
    T m
) {
    // The backward pass computes:
    //   grad_u = grad * dnd/du
    //   grad_m = grad * dnd/dm
    //
    // We need gradients of (grad_u, grad_m) w.r.t. (grad, u, m)
    //
    // d(grad_u)/d(grad) = dnd/du
    // d(grad_u)/du = grad * d^2(nd)/du^2
    // d(grad_u)/dm = grad * d^2(nd)/du/dm
    //
    // d(grad_m)/d(grad) = dnd/dm
    // d(grad_m)/du = grad * d^2(nd)/dm/du
    // d(grad_m)/dm = grad * d^2(nd)/dm^2

    T dnd_du = detail::compute_dnd_du(u, m);
    T dnd_dm = detail::compute_dnd_dm(u, m);

    T d2nd_du2 = detail::compute_d2nd_du2(u, m);
    T d2nd_dm2 = detail::compute_d2nd_dm2(u, m);
    T d2nd_dudm = detail::compute_d2nd_dudm(u, m);

    // Gradient w.r.t. grad
    T grad_grad = gg_u * dnd_du + gg_m * dnd_dm;

    // Gradient w.r.t. u
    T grad_u_out = gg_u * grad * d2nd_du2 + gg_m * grad * d2nd_dudm;

    // Gradient w.r.t. m
    T grad_m_out = gg_u * grad * d2nd_dudm + gg_m * grad * d2nd_dm2;

    return {grad_grad, grad_u_out, grad_m_out};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> jacobi_elliptic_nd_backward_backward(
    c10::complex<T> gg_u,
    c10::complex<T> gg_m,
    c10::complex<T> grad,
    c10::complex<T> u,
    c10::complex<T> m
) {
    c10::complex<T> dnd_du = detail::compute_dnd_du(u, m);
    c10::complex<T> dnd_dm = detail::compute_dnd_dm(u, m);

    c10::complex<T> d2nd_du2 = detail::compute_d2nd_du2(u, m);
    c10::complex<T> d2nd_dm2 = detail::compute_d2nd_dm2(u, m);
    c10::complex<T> d2nd_dudm = detail::compute_d2nd_dudm(u, m);

    // For complex, use Wirtinger derivatives (conjugate the derivative)
    c10::complex<T> grad_grad = gg_u * std::conj(dnd_du) + gg_m * std::conj(dnd_dm);

    c10::complex<T> grad_u_out = gg_u * grad * std::conj(d2nd_du2) + gg_m * grad * std::conj(d2nd_dudm);

    c10::complex<T> grad_m_out = gg_u * grad * std::conj(d2nd_dudm) + gg_m * grad * std::conj(d2nd_dm2);

    return {grad_grad, grad_u_out, grad_m_out};
}

} // namespace torchscience::kernel::special_functions
