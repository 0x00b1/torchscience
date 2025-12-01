#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "jacobi_elliptic_sc.h"
#include "jacobi_elliptic_sc_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Jacobi elliptic function sc(u, m)
//
// Uses numerical finite differences for second derivatives.
// Returns gradients w.r.t. (grad_output, u, m)

template <typename T>
std::tuple<T, T, T> jacobi_elliptic_sc_backward_backward(
    T grad_grad_u,
    T grad_grad_m,
    T gradient,
    T u,
    T m
) {
    // Use central finite differences for both first and second derivatives
    T h = detail::jacobi_sc_finite_diff_step<T>();

    // Compute function values at various points
    T f_center = jacobi_elliptic_sc(u, m);
    T f_u_plus = jacobi_elliptic_sc(u + h, m);
    T f_u_minus = jacobi_elliptic_sc(u - h, m);
    T f_m_plus = jacobi_elliptic_sc(u, m + h);
    T f_m_minus = jacobi_elliptic_sc(u, m - h);

    // For mixed and second partials
    T f_u_plus_m_plus = jacobi_elliptic_sc(u + h, m + h);
    T f_u_plus_m_minus = jacobi_elliptic_sc(u + h, m - h);
    T f_u_minus_m_plus = jacobi_elliptic_sc(u - h, m + h);
    T f_u_minus_m_minus = jacobi_elliptic_sc(u - h, m - h);

    // First derivatives
    T d_u = (f_u_plus - f_u_minus) / (T(2) * h);
    T d_m = (f_m_plus - f_m_minus) / (T(2) * h);

    // Second derivatives
    // d^2sc/du^2
    T d2_u_u = (f_u_plus - T(2) * f_center + f_u_minus) / (h * h);

    // d^2sc/dm^2
    T d2_m_m = (f_m_plus - T(2) * f_center + f_m_minus) / (h * h);

    // d^2sc/du dm (mixed partial)
    T d2_u_m = (f_u_plus_m_plus - f_u_plus_m_minus - f_u_minus_m_plus + f_u_minus_m_minus) / (T(4) * h * h);

    // grad_gradient: derivative of backward w.r.t. gradient
    // backward returns (gradient * d_u, gradient * d_m)
    // So d(backward)/d(gradient) = (d_u, d_m), and we project with (grad_grad_u, grad_grad_m)
    T grad_gradient = grad_grad_u * d_u + grad_grad_m * d_m;

    // grad_u: derivative of backward w.r.t. u
    // d(gradient * d_u)/du = gradient * d2_u_u
    // d(gradient * d_m)/du = gradient * d2_u_m
    T grad_u = grad_grad_u * gradient * d2_u_u + grad_grad_m * gradient * d2_u_m;

    // grad_m: derivative of backward w.r.t. m
    // d(gradient * d_u)/dm = gradient * d2_u_m
    // d(gradient * d_m)/dm = gradient * d2_m_m
    T grad_m = grad_grad_u * gradient * d2_u_m + grad_grad_m * gradient * d2_m_m;

    return {grad_gradient, grad_u, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
jacobi_elliptic_sc_backward_backward(
    c10::complex<T> grad_grad_u,
    c10::complex<T> grad_grad_m,
    c10::complex<T> gradient,
    c10::complex<T> u,
    c10::complex<T> m
) {
    T h = detail::jacobi_sc_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Compute function values
    c10::complex<T> f_center = jacobi_elliptic_sc(u, m);
    c10::complex<T> f_u_plus = jacobi_elliptic_sc(u + h_complex, m);
    c10::complex<T> f_u_minus = jacobi_elliptic_sc(u - h_complex, m);
    c10::complex<T> f_m_plus = jacobi_elliptic_sc(u, m + h_complex);
    c10::complex<T> f_m_minus = jacobi_elliptic_sc(u, m - h_complex);

    // For mixed and second partials
    c10::complex<T> f_u_plus_m_plus = jacobi_elliptic_sc(u + h_complex, m + h_complex);
    c10::complex<T> f_u_plus_m_minus = jacobi_elliptic_sc(u + h_complex, m - h_complex);
    c10::complex<T> f_u_minus_m_plus = jacobi_elliptic_sc(u - h_complex, m + h_complex);
    c10::complex<T> f_u_minus_m_minus = jacobi_elliptic_sc(u - h_complex, m - h_complex);

    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    // First derivatives
    c10::complex<T> d_u = (f_u_plus - f_u_minus) / (two * h_complex);
    c10::complex<T> d_m = (f_m_plus - f_m_minus) / (two * h_complex);

    // Second derivatives
    c10::complex<T> d2_u_u = (f_u_plus - two * f_center + f_u_minus) / (h * h);
    c10::complex<T> d2_m_m = (f_m_plus - two * f_center + f_m_minus) / (h * h);
    c10::complex<T> d2_u_m = (f_u_plus_m_plus - f_u_plus_m_minus - f_u_minus_m_plus + f_u_minus_m_minus) / (four * h * h);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    c10::complex<T> grad_gradient = grad_grad_u * std::conj(d_u) + grad_grad_m * std::conj(d_m);
    c10::complex<T> grad_u = grad_grad_u * gradient * std::conj(d2_u_u) + grad_grad_m * gradient * std::conj(d2_u_m);
    c10::complex<T> grad_m = grad_grad_u * gradient * std::conj(d2_u_m) + grad_grad_m * gradient * std::conj(d2_m_m);

    return {grad_gradient, grad_u, grad_m};
}

} // namespace torchscience::kernel::special_functions
