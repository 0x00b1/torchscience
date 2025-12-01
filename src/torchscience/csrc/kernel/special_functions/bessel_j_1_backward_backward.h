#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "bessel_j_0.h"
#include "bessel_j_1.h"
#include "bessel_j_1_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
template <typename T>
std::tuple<T, T> bessel_j_1_backward_backward(T gg_z, T grad_output, T z) {
    T j0 = bessel_j_0(z);
    T j1 = bessel_j_1(z);

    // d(backward)/d(grad_output) = J₀(z) - J₁(z)/z
    T first_deriv;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        first_deriv = T(0.5);
    } else {
        first_deriv = j0 - j1 / z;
    }
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d²J₁/dz²
    // d²J₁/dz² = -J₁(z) - J₀(z)/z + 2·J₁(z)/z²
    // Equivalently: J₁(z)·(2/z² - 1) - J₀(z)/z
    T d2_j1;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        // Use series expansion for limit:
        // J₁(z) = z/2 - z³/16 + O(z⁵), J₀(z) = 1 - z²/4 + O(z⁴)
        // d²J₁/dz² at z=0:
        // -J₁ - J₀/z + 2J₁/z² = -z/2 - 1/z + 1/z + O(z) = -z/2 + O(z) → 0
        // Equivalently, from J₁(z) series: J₁''(0) = 0
        d2_j1 = T(0);
    } else {
        T z2 = z * z;
        d2_j1 = -j1 - j0 / z + T(2) * j1 / z2;
    }
    T grad_z = gg_z * grad_output * d2_j1;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_j_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> j0 = bessel_j_0(z);
    c10::complex<T> j1 = bessel_j_1(z);

    c10::complex<T> first_deriv;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        first_deriv = c10::complex<T>(T(0.5), T(0));
    } else {
        first_deriv = j0 - j1 / z;
    }
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    c10::complex<T> d2_j1;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        // J₁''(0) = 0 (see real version for derivation)
        d2_j1 = c10::complex<T>(T(0), T(0));
    } else {
        c10::complex<T> z2 = z * z;
        d2_j1 = -j1 - j0 / z + c10::complex<T>(T(2), T(0)) * j1 / z2;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_j1);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
