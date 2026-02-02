#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include "struve_l_0.h"
#include "struve_l_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T struve_l_1_backward_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_l_1_backward_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_1_backward_zero_tolerance<double>() { return 1e-12; }

} // namespace detail

// Real backward: d/dz L_1(z) = L_0(z) - L_1(z)/z
// This is the SAME formula as H_1: d/dz H_1(z) = H_0(z) - H_1(z)/z
// The modified Struve L_1 follows the same recurrence pattern.
// At z=0: limit is 0
template <typename T>
T struve_l_1_backward(T grad_output, T z) {
    T derivative;

    if (std::abs(z) < detail::struve_l_1_backward_zero_tolerance<T>()) {
        // Limit as z -> 0: d/dz L_1(z) = 0
        // From series expansion:
        // L_1(z) = (4/(3*pi^2))*z^2 + O(z^4)
        // L_0(z) = (4/pi^2)*z + O(z^3)
        // L_1(z)/z = (4/(3*pi^2))*z + O(z^3)
        // L_0(z) - L_1(z)/z = (4/pi^2 - 4/(3*pi^2))*z + O(z^3)
        //                   = (8/(3*pi^2))*z + O(z^3)
        // The limit as z->0 is 0.
        derivative = T(0);
    } else {
        T l0 = struve_l_0(z);
        T l1 = struve_l_1(z);
        derivative = l0 - l1 / z;
    }

    return grad_output * derivative;
}

// Complex backward
template <typename T>
c10::complex<T> struve_l_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> derivative;

    if (std::abs(z) < detail::struve_l_1_backward_zero_tolerance<T>()) {
        // Limit as z -> 0: see real version for derivation
        derivative = c10::complex<T>(T(0), T(0));
    } else {
        c10::complex<T> l0 = struve_l_0(z);
        c10::complex<T> l1 = struve_l_1(z);
        derivative = l0 - l1 / z;
    }

    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
