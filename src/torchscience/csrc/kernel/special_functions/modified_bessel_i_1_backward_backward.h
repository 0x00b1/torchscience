#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "modified_bessel_i_0.h"
#include "modified_bessel_i_1.h"
#include "modified_bessel_i_1_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
template <typename T>
std::tuple<T, T> modified_bessel_i_1_backward_backward(T gg_z, T grad_output, T z) {
    T i0 = modified_bessel_i_0(z);
    T i1 = modified_bessel_i_1(z);

    // d(backward)/d(grad_output) = I₀(z) - I₁(z)/z
    T first_deriv;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        first_deriv = T(0.5);
    } else {
        first_deriv = i0 - i1 / z;
    }
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d²I₁/dz²
    // d²I₁/dz² = I₁(z) - I₀(z)/z + 2·I₁(z)/z²
    T d2_i1;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        // Use series expansion for limit:
        // I₁(z) = z/2 + z³/16 + O(z⁵), I₀(z) = 1 + z²/4 + O(z⁴)
        // I₁ - I₀/z + 2I₁/z² = z/2 - 1/z + 1/z + O(z) = z/2 + O(z) → 0
        // Equivalently, from I₁(z) series: I₁''(z) = 3z/8 + O(z³) → 0
        d2_i1 = T(0);
    } else {
        T z2 = z * z;
        d2_i1 = i1 - i0 / z + T(2) * i1 / z2;
    }
    T grad_z = gg_z * grad_output * d2_i1;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> modified_bessel_i_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> i0 = modified_bessel_i_0(z);
    c10::complex<T> i1 = modified_bessel_i_1(z);

    c10::complex<T> first_deriv;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        first_deriv = c10::complex<T>(T(0.5), T(0));
    } else {
        first_deriv = i0 - i1 / z;
    }
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    c10::complex<T> d2_i1;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        // I₁''(0) = 0 (see real version for derivation)
        d2_i1 = c10::complex<T>(T(0), T(0));
    } else {
        c10::complex<T> z2 = z * z;
        d2_i1 = i1 - i0 / z + c10::complex<T>(T(2), T(0)) * i1 / z2;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_i1);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
