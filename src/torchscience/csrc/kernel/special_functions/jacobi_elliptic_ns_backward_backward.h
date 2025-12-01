#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_ns.h"
#include "jacobi_elliptic_ns_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Jacobi elliptic function ns(u, m)
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
T compute_d2ns_du2(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));

    // Central difference: d^2(ns)/du^2 = (ns(u+h) - 2*ns(u) + ns(u-h)) / h^2
    T ns_plus = jacobi_elliptic_ns(u + h, m);
    T ns_center = jacobi_elliptic_ns(u, m);
    T ns_minus = jacobi_elliptic_ns(u - h, m);

    return (ns_plus - T(2) * ns_center + ns_minus) / (h * h);
}

template <typename T>
T compute_d2ns_dm2(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Handle boundary cases
    if (m < h) {
        // Forward difference near m = 0
        T ns_0 = jacobi_elliptic_ns(u, m);
        T ns_1 = jacobi_elliptic_ns(u, m + h);
        T ns_2 = jacobi_elliptic_ns(u, m + T(2) * h);
        return (ns_2 - T(2) * ns_1 + ns_0) / (h * h);
    } else if (m > T(1) - h) {
        // Backward difference near m = 1
        T ns_0 = jacobi_elliptic_ns(u, m);
        T ns_1 = jacobi_elliptic_ns(u, m - h);
        T ns_2 = jacobi_elliptic_ns(u, m - T(2) * h);
        return (ns_2 - T(2) * ns_1 + ns_0) / (h * h);
    }

    // Central difference
    T ns_plus = jacobi_elliptic_ns(u, m + h);
    T ns_center = jacobi_elliptic_ns(u, m);
    T ns_minus = jacobi_elliptic_ns(u, m - h);

    return (ns_plus - T(2) * ns_center + ns_minus) / (h * h);
}

template <typename T>
T compute_d2ns_dudm(T u, T m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/4.0));
    const T eps = std::numeric_limits<T>::epsilon() * T(10);

    // Mixed partial: d^2(ns)/du/dm
    // Use central differences: (ns(u+h,m+h) - ns(u+h,m-h) - ns(u-h,m+h) + ns(u-h,m-h)) / (4*h^2)

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

    T f_pp = jacobi_elliptic_ns(u + h, m_plus);
    T f_pm = jacobi_elliptic_ns(u + h, m_minus);
    T f_mp = jacobi_elliptic_ns(u - h, m_plus);
    T f_mm = jacobi_elliptic_ns(u - h, m_minus);

    T h_m = m_plus - m_minus;
    return (f_pp - f_pm - f_mp + f_mm) / (T(4) * h * h_m / T(2));
}

template <typename T>
c10::complex<T> compute_d2ns_du2(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> ns_plus = jacobi_elliptic_ns(u + h_c, m);
    c10::complex<T> ns_center = jacobi_elliptic_ns(u, m);
    c10::complex<T> ns_minus = jacobi_elliptic_ns(u - h_c, m);

    return (ns_plus - two * ns_center + ns_minus) / (h_c * h_c);
}

template <typename T>
c10::complex<T> compute_d2ns_dm2(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> two(T(2), T(0));

    c10::complex<T> ns_plus = jacobi_elliptic_ns(u, m + h_c);
    c10::complex<T> ns_center = jacobi_elliptic_ns(u, m);
    c10::complex<T> ns_minus = jacobi_elliptic_ns(u, m - h_c);

    return (ns_plus - two * ns_center + ns_minus) / (h_c * h_c);
}

template <typename T>
c10::complex<T> compute_d2ns_dudm(c10::complex<T> u, c10::complex<T> m) {
    const T h = std::pow(std::numeric_limits<T>::epsilon(), T(1.0/4.0));
    c10::complex<T> h_c(h, T(0));
    c10::complex<T> four(T(4), T(0));

    c10::complex<T> f_pp = jacobi_elliptic_ns(u + h_c, m + h_c);
    c10::complex<T> f_pm = jacobi_elliptic_ns(u + h_c, m - h_c);
    c10::complex<T> f_mp = jacobi_elliptic_ns(u - h_c, m + h_c);
    c10::complex<T> f_mm = jacobi_elliptic_ns(u - h_c, m - h_c);

    return (f_pp - f_pm - f_mp + f_mm) / (four * h_c * h_c);
}

// Compute dns/du
template <typename T>
T compute_dns_du(T u, T m) {
    auto [grad_u, grad_m] = jacobi_elliptic_ns_backward(T(1), u, m);
    return grad_u;
}

// Compute dns/dm
template <typename T>
T compute_dns_dm(T u, T m) {
    auto [grad_u, grad_m] = jacobi_elliptic_ns_backward(T(1), u, m);
    return grad_m;
}

template <typename T>
c10::complex<T> compute_dns_du(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> one(T(1), T(0));
    auto [grad_u, grad_m] = jacobi_elliptic_ns_backward(one, u, m);
    return grad_u;
}

template <typename T>
c10::complex<T> compute_dns_dm(c10::complex<T> u, c10::complex<T> m) {
    c10::complex<T> one(T(1), T(0));
    auto [grad_u, grad_m] = jacobi_elliptic_ns_backward(one, u, m);
    return grad_m;
}

} // namespace detail

template <typename T>
std::tuple<T, T, T> jacobi_elliptic_ns_backward_backward(
    T gg_u,
    T gg_m,
    T grad,
    T u,
    T m
) {
    // The backward pass computes:
    //   grad_u = grad * dns/du
    //   grad_m = grad * dns/dm
    //
    // We need gradients of (grad_u, grad_m) w.r.t. (grad, u, m)
    //
    // d(grad_u)/d(grad) = dns/du
    // d(grad_u)/du = grad * d^2(ns)/du^2
    // d(grad_u)/dm = grad * d^2(ns)/du/dm
    //
    // d(grad_m)/d(grad) = dns/dm
    // d(grad_m)/du = grad * d^2(ns)/dm/du
    // d(grad_m)/dm = grad * d^2(ns)/dm^2

    T dns_du = detail::compute_dns_du(u, m);
    T dns_dm = detail::compute_dns_dm(u, m);

    T d2ns_du2 = detail::compute_d2ns_du2(u, m);
    T d2ns_dm2 = detail::compute_d2ns_dm2(u, m);
    T d2ns_dudm = detail::compute_d2ns_dudm(u, m);

    // Gradient w.r.t. grad
    T grad_grad = gg_u * dns_du + gg_m * dns_dm;

    // Gradient w.r.t. u
    T grad_u_out = gg_u * grad * d2ns_du2 + gg_m * grad * d2ns_dudm;

    // Gradient w.r.t. m
    T grad_m_out = gg_u * grad * d2ns_dudm + gg_m * grad * d2ns_dm2;

    return {grad_grad, grad_u_out, grad_m_out};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> jacobi_elliptic_ns_backward_backward(
    c10::complex<T> gg_u,
    c10::complex<T> gg_m,
    c10::complex<T> grad,
    c10::complex<T> u,
    c10::complex<T> m
) {
    c10::complex<T> dns_du = detail::compute_dns_du(u, m);
    c10::complex<T> dns_dm = detail::compute_dns_dm(u, m);

    c10::complex<T> d2ns_du2 = detail::compute_d2ns_du2(u, m);
    c10::complex<T> d2ns_dm2 = detail::compute_d2ns_dm2(u, m);
    c10::complex<T> d2ns_dudm = detail::compute_d2ns_dudm(u, m);

    // For complex, use Wirtinger derivatives (conjugate the derivative)
    c10::complex<T> grad_grad = gg_u * std::conj(dns_du) + gg_m * std::conj(dns_dm);

    c10::complex<T> grad_u_out = gg_u * grad * std::conj(d2ns_du2) + gg_m * grad * std::conj(d2ns_dudm);

    c10::complex<T> grad_m_out = gg_u * grad * std::conj(d2ns_dudm) + gg_m * grad * std::conj(d2ns_dm2);

    return {grad_grad, grad_u_out, grad_m_out};
}

} // namespace torchscience::kernel::special_functions
