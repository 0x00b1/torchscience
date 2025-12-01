#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "bessel_j_0.h"
#include "bessel_j_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T bessel_j_zero_tolerance() {
    return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_j_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_j_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz J₁(z) = J₀(z) - J₁(z)/z
template <typename T>
T bessel_j_1_backward(T grad_output, T z) {
    T j0 = bessel_j_0(z);
    T j1 = bessel_j_1(z);

    // Handle z=0 specially: limit of J₁(z)/z as z→0 is 1/2
    // Use Taylor expansion: J₁(z) ≈ z/2 + O(z³), so J₁(z)/z → 1/2
    T derivative;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        derivative = T(0.5);  // J₀(0) - lim[J₁(z)/z] = 1 - 0.5 = 0.5
    } else {
        derivative = j0 - j1 / z;
    }
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> bessel_j_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> j0 = bessel_j_0(z);
    c10::complex<T> j1 = bessel_j_1(z);

    c10::complex<T> derivative;
    if (std::abs(z) < detail::bessel_j_zero_tolerance<T>()) {
        derivative = c10::complex<T>(T(0.5), T(0));
    } else {
        derivative = j0 - j1 / z;
    }
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
