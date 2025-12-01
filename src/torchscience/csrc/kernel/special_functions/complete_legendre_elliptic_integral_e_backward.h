#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "complete_legendre_elliptic_integral_e.h"

namespace torchscience::kernel::special_functions {

// Backward pass for complete elliptic integral of the second kind E(m)
//
// The analytical derivative is:
// dE/dm = [E(m) - K(m)] / (2m)
//
// where K(m) is the complete elliptic integral of the first kind.
//
// For simplicity and to avoid circular dependencies with K(m),
// we use numerical differentiation via central finite differences.

namespace detail {

template <typename T>
T finite_diff_step() {
    // Step size for finite differences, scaled to precision
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
T complete_legendre_elliptic_integral_e_backward(T gradient, T m) {
    // Use central finite differences: f'(m) ≈ [f(m+h) - f(m-h)] / (2h)
    T h = detail::finite_diff_step<T>();

    T f_plus = complete_legendre_elliptic_integral_e(m + h);
    T f_minus = complete_legendre_elliptic_integral_e(m - h);

    T derivative = (f_plus - f_minus) / (T(2) * h);

    return gradient * derivative;
}

template <typename T>
c10::complex<T> complete_legendre_elliptic_integral_e_backward(
    c10::complex<T> gradient,
    c10::complex<T> m
) {
    // For complex, use finite differences in real direction
    // This follows the Wirtinger derivative convention
    T h = detail::finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    c10::complex<T> f_plus = complete_legendre_elliptic_integral_e(m + h_complex);
    c10::complex<T> f_minus = complete_legendre_elliptic_integral_e(m - h_complex);

    c10::complex<T> derivative = (f_plus - f_minus) / (T(2) * h_complex);

    // PyTorch convention for holomorphic functions: grad * conj(derivative)
    return gradient * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
