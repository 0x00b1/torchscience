#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "bessel_j_0.h"
#include "bessel_j_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T bessel_j0_zero_tolerance() {
    return T(1e-12);  // Default for low-precision types
}

template <>
inline float bessel_j0_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double bessel_j0_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz J₀(z) = -J₁(z)
template <typename T>
T bessel_j_0_backward(T grad_output, T z) {
    T j1 = bessel_j_1(z);
    return grad_output * (-j1);
}

// Complex backward
template <typename T>
c10::complex<T> bessel_j_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> j1 = bessel_j_1(z);
    return grad_output * std::conj(-j1);
}

} // namespace torchscience::kernel::special_functions
