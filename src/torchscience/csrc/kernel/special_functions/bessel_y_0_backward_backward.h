#pragma once

#include <c10/util/complex.h>
#include <tuple>
#include "bessel_y_0.h"
#include "bessel_y_1.h"
#include "bessel_y_0_backward.h"
#include "bessel_y_1_backward.h"  // For bessel_y_zero_tolerance

namespace torchscience::kernel::special_functions {

// Real backward_backward
// Returns gradients for (grad_output, z)
// First derivative: d/dz Y₀(z) = -Y₁(z)
// Second derivative: d²/dz² Y₀(z) = -Y₁'(z) = -(Y₀(z) - Y₁(z)/z) = -Y₀(z) + Y₁(z)/z
template <typename T>
std::tuple<T, T> bessel_y_0_backward_backward(T gg_z, T grad_output, T z) {
    // Y₀ is only defined for z > 0
    if (z <= T(0)) {
        return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
    }

    T y0 = bessel_y_0(z);
    T y1 = bessel_y_1(z);

    // d(backward)/d(grad_output) = -Y₁(z)
    T first_deriv = -y1;
    T grad_grad_output = gg_z * first_deriv;

    // d(backward)/dz = grad_output * d²Y₀/dz²
    // d²Y₀/dz² = -Y₀(z) + Y₁(z)/z
    T d2_y0;
    if (z < detail::bessel_y_zero_tolerance<T>()) {
        // Near z=0+:
        // Y₀(z) ~ (2/π)ln(z/2), Y₁(z) ~ -2/(πz)
        // Y₁(z)/z ~ -2/(πz²)
        // d²Y₀/dz² ~ -2/(πz²) (dominated by Y₁/z term)
        d2_y0 = T(-2) / (T(3.14159265358979323846) * z * z);
    } else {
        d2_y0 = -y0 + y1 / z;
    }
    T grad_z = gg_z * grad_output * d2_y0;

    return {grad_grad_output, grad_z};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> bessel_y_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y0 = bessel_y_0(z);
    c10::complex<T> y1 = bessel_y_1(z);

    // d(backward)/d(grad_output) = -Y₁(z)
    c10::complex<T> first_deriv = -y1;
    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);

    // d²Y₀/dz² = -Y₀(z) + Y₁(z)/z
    c10::complex<T> d2_y0;
    if (std::abs(z) < detail::bessel_y_zero_tolerance<T>()) {
        c10::complex<T> pi_val = c10::complex<T>(T(3.14159265358979323846), T(0));
        d2_y0 = c10::complex<T>(T(-2), T(0)) / (pi_val * z * z);
    } else {
        d2_y0 = -y0 + y1 / z;
    }
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(d2_y0);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
