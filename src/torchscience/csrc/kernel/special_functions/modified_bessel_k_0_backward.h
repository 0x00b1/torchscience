#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "modified_bessel_k_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T bessel_k_zero_tolerance() {
    return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_k_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_k_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz K_0(z) = -K_1(z)
template <typename T>
T modified_bessel_k_0_backward(T grad_output, T z) {
    // K_0 is only defined for z > 0
    if (z <= T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T k1 = modified_bessel_k_1(z);
    T derivative = -k1;
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> modified_bessel_k_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> k1 = modified_bessel_k_1(z);
    c10::complex<T> derivative = -k1;
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
