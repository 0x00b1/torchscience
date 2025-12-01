#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "complete_legendre_elliptic_integral_e.h"
#include "complete_legendre_elliptic_integral_e_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for complete elliptic integral of the second kind E(m)
//
// Uses numerical finite differences for the second derivative.
// Returns gradients w.r.t. (grad_output, m)

template <typename T>
std::tuple<T, T> complete_legendre_elliptic_integral_e_backward_backward(
    T grad_grad_m,
    T gradient,
    T m
) {
    // Use central finite differences for both first and second derivatives
    T h = detail::finite_diff_step<T>();

    // First derivative for grad_gradient (same as forward derivative)
    T f_plus = complete_legendre_elliptic_integral_e(m + h);
    T f_minus = complete_legendre_elliptic_integral_e(m - h);
    T f_m = complete_legendre_elliptic_integral_e(m);

    T first_derivative = (f_plus - f_minus) / (T(2) * h);

    // Second derivative: f''(m) ≈ [f(m+h) - 2f(m) + f(m-h)] / h^2
    T second_derivative = (f_plus - T(2) * f_m + f_minus) / (h * h);

    // grad_gradient: derivative of backward w.r.t. gradient = first_derivative
    T grad_gradient = grad_grad_m * first_derivative;

    // grad_m: derivative of backward w.r.t. m = gradient * second_derivative
    T grad_m = grad_grad_m * gradient * second_derivative;

    return {grad_gradient, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> complete_legendre_elliptic_integral_e_backward_backward(
    c10::complex<T> grad_grad_m,
    c10::complex<T> gradient,
    c10::complex<T> m
) {
    T h = detail::finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    c10::complex<T> f_plus = complete_legendre_elliptic_integral_e(m + h_complex);
    c10::complex<T> f_minus = complete_legendre_elliptic_integral_e(m - h_complex);
    c10::complex<T> f_m = complete_legendre_elliptic_integral_e(m);

    c10::complex<T> first_derivative = (f_plus - f_minus) / (T(2) * h_complex);
    c10::complex<T> second_derivative = (f_plus - c10::complex<T>(T(2), T(0)) * f_m + f_minus) / (h * h);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    c10::complex<T> grad_gradient = grad_grad_m * std::conj(first_derivative);
    c10::complex<T> grad_m = grad_grad_m * gradient * std::conj(second_derivative);

    return {grad_gradient, grad_m};
}

} // namespace torchscience::kernel::special_functions
