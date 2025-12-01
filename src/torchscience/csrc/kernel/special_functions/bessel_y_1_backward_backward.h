#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "bessel_y_0.h"
#include "bessel_y_1.h"
#include "bessel_y_1_backward.h"

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
// First derivative: d/dz Y₁(z) = Y₀(z) - Y₁(z)/z
// Second derivative: d²/dz² Y₁(z) = -Y₁(z) - Y₀(z)/z + 2Y₁(z)/z²
// Derived from:
//   d/dz [Y₀(z) - Y₁(z)/z]
//   = Y₀'(z) - [Y₁'(z)/z - Y₁(z)/z²]
//   = -Y₁(z) - [Y₀(z) - Y₁(z)/z]/z + Y₁(z)/z²
//   = -Y₁(z) - Y₀(z)/z + Y₁(z)/z² + Y₁(z)/z²
//   = -Y₁(z) - Y₀(z)/z + 2Y₁(z)/z²
template <typename T>
std::tuple<T, T> bessel_y_1_backward_backward(T gg_z, T grad_output, T z) {
    // Y₁ is only defined for z > 0
    if (z <= T(0)) {
        return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
    }

    T y0 = bessel_y_0(z);
    T y1 = bessel_y_1(z);

    // d(backward)/d(grad_output) = Y₀(z) - Y₁(z)/z
    T first_deriv;
    if (z < detail::bessel_y_zero_tolerance<T>()) {
        // Near z=0+, the derivative diverges
        first_deriv = T(2) / (T(3.14159265358979323846) * z * z);
    } else {
        first_deriv = y0 - y1 / z;
    }
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d²Y₁/dz²
    // d²Y₁/dz² = -Y₁(z) - Y₀(z)/z + 2·Y₁(z)/z²
    // Equivalently: Y₁(z)·(2/z² - 1) - Y₀(z)/z
    T d2_y1;
    if (z < detail::bessel_y_zero_tolerance<T>()) {
        // Near z=0+:
        // Y₀(z) ~ (2/π)ln(z/2) + γ terms
        // Y₁(z) ~ -2/(πz)
        // d²Y₁/dz² ~ -4/(πz³)
        d2_y1 = T(-4) / (T(3.14159265358979323846) * z * z * z);
    } else {
        T z2 = z * z;
        d2_y1 = -y1 - y0 / z + T(2) * y1 / z2;
    }
    T grad_z = gg_z * grad_output * d2_y1;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_y_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y0 = bessel_y_0(z);
    c10::complex<T> y1 = bessel_y_1(z);

    c10::complex<T> first_deriv;
    if (std::abs(z) < detail::bessel_y_zero_tolerance<T>()) {
        c10::complex<T> pi_val = c10::complex<T>(T(3.14159265358979323846), T(0));
        first_deriv = c10::complex<T>(T(2), T(0)) / (pi_val * z * z);
    } else {
        first_deriv = y0 - y1 / z;
    }
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    c10::complex<T> d2_y1;
    if (std::abs(z) < detail::bessel_y_zero_tolerance<T>()) {
        c10::complex<T> pi_val = c10::complex<T>(T(3.14159265358979323846), T(0));
        d2_y1 = c10::complex<T>(T(-4), T(0)) / (pi_val * z * z * z);
    } else {
        c10::complex<T> z2 = z * z;
        d2_y1 = -y1 - y0 / z + c10::complex<T>(T(2), T(0)) * y1 / z2;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_y1);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
