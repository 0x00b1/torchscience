#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "inverse_jacobi_elliptic_sd.h"
#include "inverse_jacobi_elliptic_sd_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for inverse Jacobi elliptic function arcsd(x, m)
//
// Uses numerical finite differences for second derivatives.
// Returns gradients w.r.t. (grad_output, x, m)

template <typename T>
std::tuple<T, T, T> inverse_jacobi_elliptic_sd_backward_backward(
    T grad_grad_x,
    T grad_grad_m,
    T gradient,
    T x,
    T m
) {
    // Use central finite differences for both first and second derivatives
    T h = detail::inverse_jacobi_sd_finite_diff_step<T>();

    // Compute function values at various points
    T f_center = inverse_jacobi_elliptic_sd(x, m);
    T f_x_plus = inverse_jacobi_elliptic_sd(x + h, m);
    T f_x_minus = inverse_jacobi_elliptic_sd(x - h, m);
    T f_m_plus = inverse_jacobi_elliptic_sd(x, m + h);
    T f_m_minus = inverse_jacobi_elliptic_sd(x, m - h);

    // For mixed and second partials
    T f_x_plus_m_plus = inverse_jacobi_elliptic_sd(x + h, m + h);
    T f_x_plus_m_minus = inverse_jacobi_elliptic_sd(x + h, m - h);
    T f_x_minus_m_plus = inverse_jacobi_elliptic_sd(x - h, m + h);
    T f_x_minus_m_minus = inverse_jacobi_elliptic_sd(x - h, m - h);

    // First derivatives
    T d_x = (f_x_plus - f_x_minus) / (T(2) * h);
    T d_m = (f_m_plus - f_m_minus) / (T(2) * h);

    // Second derivatives
    // d^2f/dx^2
    T d2_x_x = (f_x_plus - T(2) * f_center + f_x_minus) / (h * h);

    // d^2f/dm^2
    T d2_m_m = (f_m_plus - T(2) * f_center + f_m_minus) / (h * h);

    // d^2f/dx dm (mixed partial)
    T d2_x_m = (f_x_plus_m_plus - f_x_plus_m_minus - f_x_minus_m_plus + f_x_minus_m_minus) / (T(4) * h * h);

    // grad_gradient: derivative of backward w.r.t. gradient
    // backward returns (gradient * d_x, gradient * d_m)
    // So d(backward)/d(gradient) = (d_x, d_m), and we project with (grad_grad_x, grad_grad_m)
    T grad_gradient = grad_grad_x * d_x + grad_grad_m * d_m;

    // grad_x: derivative of backward w.r.t. x
    // d(gradient * d_x)/dx = gradient * d2_x_x
    // d(gradient * d_m)/dx = gradient * d2_x_m
    T grad_x = grad_grad_x * gradient * d2_x_x + grad_grad_m * gradient * d2_x_m;

    // grad_m: derivative of backward w.r.t. m
    // d(gradient * d_x)/dm = gradient * d2_x_m
    // d(gradient * d_m)/dm = gradient * d2_m_m
    T grad_m = grad_grad_x * gradient * d2_x_m + grad_grad_m * gradient * d2_m_m;

    return {grad_gradient, grad_x, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
inverse_jacobi_elliptic_sd_backward_backward(
    c10::complex<T> grad_grad_x,
    c10::complex<T> grad_grad_m,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> m
) {
    T h = detail::inverse_jacobi_sd_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Compute function values
    c10::complex<T> f_center = inverse_jacobi_elliptic_sd(x, m);
    c10::complex<T> f_x_plus = inverse_jacobi_elliptic_sd(x + h_complex, m);
    c10::complex<T> f_x_minus = inverse_jacobi_elliptic_sd(x - h_complex, m);
    c10::complex<T> f_m_plus = inverse_jacobi_elliptic_sd(x, m + h_complex);
    c10::complex<T> f_m_minus = inverse_jacobi_elliptic_sd(x, m - h_complex);

    // For mixed and second partials
    c10::complex<T> f_x_plus_m_plus = inverse_jacobi_elliptic_sd(x + h_complex, m + h_complex);
    c10::complex<T> f_x_plus_m_minus = inverse_jacobi_elliptic_sd(x + h_complex, m - h_complex);
    c10::complex<T> f_x_minus_m_plus = inverse_jacobi_elliptic_sd(x - h_complex, m + h_complex);
    c10::complex<T> f_x_minus_m_minus = inverse_jacobi_elliptic_sd(x - h_complex, m - h_complex);

    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    // First derivatives
    c10::complex<T> d_x = (f_x_plus - f_x_minus) / (two * h_complex);
    c10::complex<T> d_m = (f_m_plus - f_m_minus) / (two * h_complex);

    // Second derivatives
    c10::complex<T> d2_x_x = (f_x_plus - two * f_center + f_x_minus) / (h * h);
    c10::complex<T> d2_m_m = (f_m_plus - two * f_center + f_m_minus) / (h * h);
    c10::complex<T> d2_x_m = (f_x_plus_m_plus - f_x_plus_m_minus - f_x_minus_m_plus + f_x_minus_m_minus) / (four * h * h);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    c10::complex<T> grad_gradient = grad_grad_x * std::conj(d_x) + grad_grad_m * std::conj(d_m);
    c10::complex<T> grad_x = grad_grad_x * gradient * std::conj(d2_x_x) + grad_grad_m * gradient * std::conj(d2_x_m);
    c10::complex<T> grad_m = grad_grad_x * gradient * std::conj(d2_x_m) + grad_grad_m * gradient * std::conj(d2_m_m);

    return {grad_gradient, grad_x, grad_m};
}

} // namespace torchscience::kernel::special_functions
