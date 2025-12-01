#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "modified_bessel_i_0.h"
#include "modified_bessel_i_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T bessel_zero_tolerance() {
  return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz I₁(z) = I₀(z) - I₁(z)/z
template <typename T>
T modified_bessel_i_1_backward(T grad_output, T z) {
    T i0 = modified_bessel_i_0(z);
    T i1 = modified_bessel_i_1(z);
    // Handle z=0 specially: limit of I₁(z)/z as z→0 is 1/2
    // Use Taylor expansion: I₁(z) ≈ z/2 + O(z³), so I₁(z)/z → 1/2
    T derivative;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        derivative = T(0.5);  // I₀(0) - lim[I₁(z)/z] = 1 - 0.5 = 0.5
    } else {
        derivative = i0 - i1 / z;
    }
    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> modified_bessel_i_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> i0 = modified_bessel_i_0(z);
    c10::complex<T> i1 = modified_bessel_i_1(z);
    c10::complex<T> derivative;
    if (std::abs(z) < detail::bessel_zero_tolerance<T>()) {
        derivative = c10::complex<T>(T(0.5), T(0));
    } else {
        derivative = i0 - i1 / z;
    }
    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
