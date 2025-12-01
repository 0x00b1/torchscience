#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "incomplete_legendre_elliptic_integral_e.h"
#include "incomplete_legendre_elliptic_integral_e_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for incomplete elliptic integral of the second kind E(phi, m)
//
// Uses numerical finite differences for second derivatives.
// Returns gradients w.r.t. (grad_output, phi, m)

template <typename T>
std::tuple<T, T, T> incomplete_legendre_elliptic_integral_e_backward_backward(
    T grad_grad_phi,
    T grad_grad_m,
    T gradient,
    T phi,
    T m
) {
    // Use central finite differences for both first and second derivatives
    T h = detail::incomplete_e_finite_diff_step<T>();

    // Compute function values at phi-h, phi, phi+h for various m values
    T f_center = incomplete_legendre_elliptic_integral_e(phi, m);
    T f_phi_plus = incomplete_legendre_elliptic_integral_e(phi + h, m);
    T f_phi_minus = incomplete_legendre_elliptic_integral_e(phi - h, m);
    T f_m_plus = incomplete_legendre_elliptic_integral_e(phi, m + h);
    T f_m_minus = incomplete_legendre_elliptic_integral_e(phi, m - h);

    // For mixed and second partials
    T f_phi_plus_m_plus = incomplete_legendre_elliptic_integral_e(phi + h, m + h);
    T f_phi_plus_m_minus = incomplete_legendre_elliptic_integral_e(phi + h, m - h);
    T f_phi_minus_m_plus = incomplete_legendre_elliptic_integral_e(phi - h, m + h);
    T f_phi_minus_m_minus = incomplete_legendre_elliptic_integral_e(phi - h, m - h);

    // First derivatives
    T d_phi = (f_phi_plus - f_phi_minus) / (T(2) * h);
    T d_m = (f_m_plus - f_m_minus) / (T(2) * h);

    // Second derivatives
    // d^2E/dphi^2
    T d2_phi_phi = (f_phi_plus - T(2) * f_center + f_phi_minus) / (h * h);

    // d^2E/dm^2
    T d2_m_m = (f_m_plus - T(2) * f_center + f_m_minus) / (h * h);

    // d^2E/dphi dm (mixed partial)
    T d2_phi_m = (f_phi_plus_m_plus - f_phi_plus_m_minus - f_phi_minus_m_plus + f_phi_minus_m_minus) / (T(4) * h * h);

    // grad_gradient: derivative of backward w.r.t. gradient
    // backward returns (gradient * d_phi, gradient * d_m)
    // So d(backward)/d(gradient) = (d_phi, d_m), and we project with (grad_grad_phi, grad_grad_m)
    T grad_gradient = grad_grad_phi * d_phi + grad_grad_m * d_m;

    // grad_phi: derivative of backward w.r.t. phi
    // d(gradient * d_phi)/dphi = gradient * d2_phi_phi
    // d(gradient * d_m)/dphi = gradient * d2_phi_m
    T grad_phi = grad_grad_phi * gradient * d2_phi_phi + grad_grad_m * gradient * d2_phi_m;

    // grad_m: derivative of backward w.r.t. m
    // d(gradient * d_phi)/dm = gradient * d2_phi_m
    // d(gradient * d_m)/dm = gradient * d2_m_m
    T grad_m = grad_grad_phi * gradient * d2_phi_m + grad_grad_m * gradient * d2_m_m;

    return {grad_gradient, grad_phi, grad_m};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
incomplete_legendre_elliptic_integral_e_backward_backward(
    c10::complex<T> grad_grad_phi,
    c10::complex<T> grad_grad_m,
    c10::complex<T> gradient,
    c10::complex<T> phi,
    c10::complex<T> m
) {
    T h = detail::incomplete_e_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Compute function values
    c10::complex<T> f_center = incomplete_legendre_elliptic_integral_e(phi, m);
    c10::complex<T> f_phi_plus = incomplete_legendre_elliptic_integral_e(phi + h_complex, m);
    c10::complex<T> f_phi_minus = incomplete_legendre_elliptic_integral_e(phi - h_complex, m);
    c10::complex<T> f_m_plus = incomplete_legendre_elliptic_integral_e(phi, m + h_complex);
    c10::complex<T> f_m_minus = incomplete_legendre_elliptic_integral_e(phi, m - h_complex);

    // For mixed and second partials
    c10::complex<T> f_phi_plus_m_plus = incomplete_legendre_elliptic_integral_e(phi + h_complex, m + h_complex);
    c10::complex<T> f_phi_plus_m_minus = incomplete_legendre_elliptic_integral_e(phi + h_complex, m - h_complex);
    c10::complex<T> f_phi_minus_m_plus = incomplete_legendre_elliptic_integral_e(phi - h_complex, m + h_complex);
    c10::complex<T> f_phi_minus_m_minus = incomplete_legendre_elliptic_integral_e(phi - h_complex, m - h_complex);

    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    // First derivatives
    c10::complex<T> d_phi = (f_phi_plus - f_phi_minus) / (two * h_complex);
    c10::complex<T> d_m = (f_m_plus - f_m_minus) / (two * h_complex);

    // Second derivatives
    c10::complex<T> d2_phi_phi = (f_phi_plus - two * f_center + f_phi_minus) / (h * h);
    c10::complex<T> d2_m_m = (f_m_plus - two * f_center + f_m_minus) / (h * h);
    c10::complex<T> d2_phi_m = (f_phi_plus_m_plus - f_phi_plus_m_minus - f_phi_minus_m_plus + f_phi_minus_m_minus) / (four * h * h);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    c10::complex<T> grad_gradient = grad_grad_phi * std::conj(d_phi) + grad_grad_m * std::conj(d_m);
    c10::complex<T> grad_phi = grad_grad_phi * gradient * std::conj(d2_phi_phi) + grad_grad_m * gradient * std::conj(d2_phi_m);
    c10::complex<T> grad_m = grad_grad_phi * gradient * std::conj(d2_phi_m) + grad_grad_m * gradient * std::conj(d2_m_m);

    return {grad_gradient, grad_phi, grad_m};
}

} // namespace torchscience::kernel::special_functions
