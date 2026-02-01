#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "weierstrass_eta.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Weierstrass eta quasi-period eta1(g2, g3)
//
// Uses numerical finite differences to compute gradients with respect to
// g2 and g3.

namespace detail {

template <typename T>
inline T weierstrass_eta_finite_diff_step_backward() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float weierstrass_eta_finite_diff_step_backward<float>() {
    return 1e-4f;
}

template <>
inline double weierstrass_eta_finite_diff_step_backward<double>() {
    return 1e-7;
}

} // namespace detail

template <typename T>
std::tuple<T, T> weierstrass_eta_backward(
    T gradient,
    T g2,
    T g3
) {
    // Use central finite differences: f'(x) = [f(x+h) - f(x-h)] / (2h)
    T h = detail::weierstrass_eta_finite_diff_step_backward<T>();

    // Gradient w.r.t. g2
    T f_g2_plus = weierstrass_eta(g2 + h, g3);
    T f_g2_minus = weierstrass_eta(g2 - h, g3);
    T derivative_g2 = (f_g2_plus - f_g2_minus) / (T(2) * h);

    // Gradient w.r.t. g3
    T f_g3_plus = weierstrass_eta(g2, g3 + h);
    T f_g3_minus = weierstrass_eta(g2, g3 - h);
    T derivative_g3 = (f_g3_plus - f_g3_minus) / (T(2) * h);

    return {gradient * derivative_g2, gradient * derivative_g3};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> weierstrass_eta_backward(
    c10::complex<T> gradient,
    c10::complex<T> g2,
    c10::complex<T> g3
) {
    // For complex, use finite differences in real direction
    // This follows the Wirtinger derivative convention
    T h = detail::weierstrass_eta_finite_diff_step_backward<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. g2
    c10::complex<T> f_g2_plus = weierstrass_eta(g2 + h_complex, g3);
    c10::complex<T> f_g2_minus = weierstrass_eta(g2 - h_complex, g3);
    c10::complex<T> derivative_g2 = (f_g2_plus - f_g2_minus) / (T(2) * h_complex);

    // Gradient w.r.t. g3
    c10::complex<T> f_g3_plus = weierstrass_eta(g2, g3 + h_complex);
    c10::complex<T> f_g3_minus = weierstrass_eta(g2, g3 - h_complex);
    c10::complex<T> derivative_g3 = (f_g3_plus - f_g3_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return {
        gradient * std::conj(derivative_g2),
        gradient * std::conj(derivative_g3)
    };
}

} // namespace torchscience::kernel::special_functions
