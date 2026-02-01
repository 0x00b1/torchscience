#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <tuple>

#include "weierstrass_eta.h"
#include "weierstrass_eta_backward.h"

namespace torchscience::kernel::special_functions {

// Second-order backward pass for Weierstrass eta quasi-period eta1(g2, g3)
//
// Uses numerical finite differences for second derivatives.
// Returns gradients w.r.t. (grad_output, g2, g3)

template <typename T>
std::tuple<T, T, T> weierstrass_eta_backward_backward(
    T grad_grad_g2,
    T grad_grad_g3,
    T gradient,
    T g2,
    T g3
) {
    // Use central finite differences for both first and second derivatives
    T h = detail::weierstrass_eta_finite_diff_step_backward<T>();

    // Compute function values at various points
    T f_center = weierstrass_eta(g2, g3);
    T f_g2_plus = weierstrass_eta(g2 + h, g3);
    T f_g2_minus = weierstrass_eta(g2 - h, g3);
    T f_g3_plus = weierstrass_eta(g2, g3 + h);
    T f_g3_minus = weierstrass_eta(g2, g3 - h);

    // For mixed partials
    T f_g2_plus_g3_plus = weierstrass_eta(g2 + h, g3 + h);
    T f_g2_plus_g3_minus = weierstrass_eta(g2 + h, g3 - h);
    T f_g2_minus_g3_plus = weierstrass_eta(g2 - h, g3 + h);
    T f_g2_minus_g3_minus = weierstrass_eta(g2 - h, g3 - h);

    // First derivatives
    T d_g2 = (f_g2_plus - f_g2_minus) / (T(2) * h);
    T d_g3 = (f_g3_plus - f_g3_minus) / (T(2) * h);

    // Second derivatives (pure)
    T d2_g2_g2 = (f_g2_plus - T(2) * f_center + f_g2_minus) / (h * h);
    T d2_g3_g3 = (f_g3_plus - T(2) * f_center + f_g3_minus) / (h * h);

    // Second derivatives (mixed partial)
    T d2_g2_g3 = (f_g2_plus_g3_plus - f_g2_plus_g3_minus - f_g2_minus_g3_plus + f_g2_minus_g3_minus) / (T(4) * h * h);

    // grad_gradient: derivative of backward w.r.t. gradient
    // backward returns (gradient * d_g2, gradient * d_g3)
    // So d(backward)/d(gradient) = (d_g2, d_g3), and we project with (grad_grad_g2, grad_grad_g3)
    T grad_gradient = grad_grad_g2 * d_g2 + grad_grad_g3 * d_g3;

    // grad_g2: derivative of backward w.r.t. g2
    // d(gradient * d_g2)/dg2 = gradient * d2_g2_g2
    // d(gradient * d_g3)/dg2 = gradient * d2_g2_g3
    T grad_g2 = grad_grad_g2 * gradient * d2_g2_g2 + grad_grad_g3 * gradient * d2_g2_g3;

    // grad_g3: derivative of backward w.r.t. g3
    // d(gradient * d_g2)/dg3 = gradient * d2_g2_g3
    // d(gradient * d_g3)/dg3 = gradient * d2_g3_g3
    T grad_g3 = grad_grad_g2 * gradient * d2_g2_g3 + grad_grad_g3 * gradient * d2_g3_g3;

    return {grad_gradient, grad_g2, grad_g3};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
weierstrass_eta_backward_backward(
    c10::complex<T> grad_grad_g2,
    c10::complex<T> grad_grad_g3,
    c10::complex<T> gradient,
    c10::complex<T> g2,
    c10::complex<T> g3
) {
    T h = detail::weierstrass_eta_finite_diff_step_backward<T>();
    c10::complex<T> h_complex(h, T(0));

    // Compute function values
    c10::complex<T> f_center = weierstrass_eta(g2, g3);
    c10::complex<T> f_g2_plus = weierstrass_eta(g2 + h_complex, g3);
    c10::complex<T> f_g2_minus = weierstrass_eta(g2 - h_complex, g3);
    c10::complex<T> f_g3_plus = weierstrass_eta(g2, g3 + h_complex);
    c10::complex<T> f_g3_minus = weierstrass_eta(g2, g3 - h_complex);

    // For mixed partials
    c10::complex<T> f_g2_plus_g3_plus = weierstrass_eta(g2 + h_complex, g3 + h_complex);
    c10::complex<T> f_g2_plus_g3_minus = weierstrass_eta(g2 + h_complex, g3 - h_complex);
    c10::complex<T> f_g2_minus_g3_plus = weierstrass_eta(g2 - h_complex, g3 + h_complex);
    c10::complex<T> f_g2_minus_g3_minus = weierstrass_eta(g2 - h_complex, g3 - h_complex);

    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));

    // First derivatives
    c10::complex<T> d_g2 = (f_g2_plus - f_g2_minus) / (two * h_complex);
    c10::complex<T> d_g3 = (f_g3_plus - f_g3_minus) / (two * h_complex);

    // Second derivatives (pure)
    c10::complex<T> d2_g2_g2 = (f_g2_plus - two * f_center + f_g2_minus) / (h * h);
    c10::complex<T> d2_g3_g3 = (f_g3_plus - two * f_center + f_g3_minus) / (h * h);

    // Second derivatives (mixed partial)
    c10::complex<T> d2_g2_g3 = (f_g2_plus_g3_plus - f_g2_plus_g3_minus - f_g2_minus_g3_plus + f_g2_minus_g3_minus) / (four * h * h);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    c10::complex<T> grad_gradient = grad_grad_g2 * std::conj(d_g2) + grad_grad_g3 * std::conj(d_g3);
    c10::complex<T> grad_g2 = grad_grad_g2 * gradient * std::conj(d2_g2_g2) + grad_grad_g3 * gradient * std::conj(d2_g2_g3);
    c10::complex<T> grad_g3 = grad_grad_g2 * gradient * std::conj(d2_g2_g3) + grad_grad_g3 * gradient * std::conj(d2_g3_g3);

    return {grad_gradient, grad_g2, grad_g3};
}

} // namespace torchscience::kernel::special_functions
