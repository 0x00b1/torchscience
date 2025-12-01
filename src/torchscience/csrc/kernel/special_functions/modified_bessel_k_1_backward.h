#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "modified_bessel_k_0.h"
#include "modified_bessel_k_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Note: bessel_k_zero_tolerance is defined in modified_bessel_k_0_backward.h
// We use inline to avoid ODR violations
template <typename T>
inline T bessel_k1_zero_tolerance() {
    return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_k1_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_k1_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz K_1(z) = -K_0(z) - K_1(z)/z
template <typename T>
T modified_bessel_k_1_backward(T grad_output, T z) {
    // K_1 is only defined for z > 0
    if (z <= T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T k0 = modified_bessel_k_0(z);
    T k1 = modified_bessel_k_1(z);

    // Handle z near 0 specially: K_1(z)/z diverges as z -> 0+
    // Near z=0+, K_1(z) ~ 1/z, so K_1(z)/z ~ 1/z^2
    // The derivative -K_0(z) - K_1(z)/z also diverges
    T derivative;
    if (z < detail::bessel_k1_zero_tolerance<T>()) {
        // Very small z: derivative diverges
        // K_0(z) ~ -ln(z/2) - gamma and K_1(z)/z ~ 1/z^2
        // The -K_1(z)/z term dominates: d/dz K_1(z) ~ -1/z^2
        derivative = -T(1) / (z * z);
    } else {
        derivative = -k0 - k1 / z;
    }
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> modified_bessel_k_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> k0 = modified_bessel_k_0(z);
    c10::complex<T> k1 = modified_bessel_k_1(z);

    c10::complex<T> derivative;
    if (std::abs(z) < detail::bessel_k1_zero_tolerance<T>()) {
        // Near z=0, derivative diverges
        derivative = -c10::complex<T>(T(1), T(0)) / (z * z);
    } else {
        derivative = -k0 - k1 / z;
    }
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
