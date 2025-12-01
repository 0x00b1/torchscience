#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_k_1.h"

namespace torchscience::kernel::special_functions {

// d/dz k_1(z) = -(pi/2z^3)(2 + z + z^2) e^(-z)
//
// Derivation:
// k_1(z) = (pi/2z^2)(1+z) e^(-z)
// Let f(z) = (1+z)/z^2 * e^(-z)
// f'(z) = d/dz[(1+z)/z^2] * e^(-z) + (1+z)/z^2 * d/dz[e^(-z)]
//       = [(z^2 - 2z(1+z))/z^4] * e^(-z) - (1+z)/z^2 * e^(-z)
//       = [(z - 2 - 2z)/z^3] * e^(-z) - (1+z)/z^2 * e^(-z)
//       = [(-z - 2)/z^3] * e^(-z) - (1+z)/z^2 * e^(-z)
//       = e^(-z) * [(-z - 2)/z^3 - (1+z)/z^2]
//       = e^(-z) * [(-z - 2 - z(1+z))/z^3]
//       = e^(-z) * [(-z - 2 - z - z^2)/z^3]
//       = e^(-z) * [(-2z - 2 - z^2)/z^3]
//       = -e^(-z) * (2 + 2z + z^2)/z^3
//       = -e^(-z) * (2 + z + z^2)/z^3  [NOTE: correcting, double-check]
//
// Actually, let me re-derive more carefully:
// k_1(z) = (pi/2) * (1+z) * z^(-2) * e^(-z)
// Using product rule: (uvw)' = u'vw + uv'w + uvw'
// u = (1+z), u' = 1
// v = z^(-2), v' = -2z^(-3)
// w = e^(-z), w' = -e^(-z)
//
// k_1'(z) = (pi/2) * [1 * z^(-2) * e^(-z) + (1+z) * (-2z^(-3)) * e^(-z) + (1+z) * z^(-2) * (-e^(-z))]
//         = (pi/2) * e^(-z) * [z^(-2) - 2(1+z)z^(-3) - (1+z)z^(-2)]
//         = (pi/2) * e^(-z) * z^(-3) * [z - 2(1+z) - (1+z)z]
//         = (pi/2) * e^(-z) * z^(-3) * [z - 2 - 2z - z - z^2]
//         = (pi/2) * e^(-z) * z^(-3) * [-2z - 2 - z^2]
//         = -(pi/2) * e^(-z) * z^(-3) * [2 + 2z + z^2]
//
// So: d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z)

template <typename T>
T spherical_bessel_k_1_backward(T grad_output, T z) {
    const T pi = T(3.14159265358979323846);
    const T eps = detail::spherical_bessel_k_1_eps<T>();

    // For small z near pole, gradient is very large negative
    if (std::abs(z) < eps) {
        return grad_output * (-std::numeric_limits<T>::infinity());
    }

    T z_inv = T(1) / z;
    T z2 = z * z;
    T z3_inv = z_inv * z_inv * z_inv;

    // d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z)
    T deriv = -(pi / T(2)) * z3_inv * (T(2) + T(2) * z + z2) * std::exp(-z);

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    const T pi = T(3.14159265358979323846);
    const T eps = detail::spherical_bessel_k_1_eps<T>();

    if (std::abs(z) < eps) {
        return grad_output * c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> z_inv = c10::complex<T>(T(1), T(0)) / z;
    c10::complex<T> z2 = z * z;
    c10::complex<T> z3_inv = z_inv * z_inv * z_inv;

    c10::complex<T> pi_over_2(pi / T(2), T(0));
    c10::complex<T> two(T(2), T(0));

    // d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z)
    c10::complex<T> deriv = -pi_over_2 * z3_inv * (two + two * z + z2) * std::exp(-z);

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
