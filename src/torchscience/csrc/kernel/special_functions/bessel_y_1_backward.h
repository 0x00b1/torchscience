#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "bessel_y_0.h"
#include "bessel_y_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T bessel_y_zero_tolerance() {
    return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_y_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_y_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz Y₁(z) = Y₀(z) - Y₁(z)/z
template <typename T>
T bessel_y_1_backward(T grad_output, T z) {
    // Y₁ is only defined for z > 0
    if (z <= T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T y0 = bessel_y_0(z);
    T y1 = bessel_y_1(z);

    // Handle z near 0 specially: Y₁(z)/z diverges as z → 0+
    // Near z=0+, Y₁(z) ~ -2/(πz), so Y₁(z)/z ~ -2/(πz²)
    // The derivative Y₀(z) - Y₁(z)/z also diverges
    T derivative;
    if (z < detail::bessel_y_zero_tolerance<T>()) {
        // Very small z: derivative is large and negative
        // Y₀(z) ~ (2/π)ln(z) and Y₁(z)/z ~ -2/(πz²)
        // The -Y₁(z)/z term dominates: d/dz Y₁(z) ~ 2/(πz²)
        derivative = T(2) / (T(3.14159265358979323846) * z * z);
    } else {
        derivative = y0 - y1 / z;
    }
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> bessel_y_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y0 = bessel_y_0(z);
    c10::complex<T> y1 = bessel_y_1(z);

    c10::complex<T> derivative;
    if (std::abs(z) < detail::bessel_y_zero_tolerance<T>()) {
        // Near z=0, derivative diverges
        c10::complex<T> pi_val = c10::complex<T>(T(3.14159265358979323846), T(0));
        derivative = c10::complex<T>(T(2), T(0)) / (pi_val * z * z);
    } else {
        derivative = y0 - y1 / z;
    }
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
