#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_k_0.h"

namespace torchscience::kernel::special_functions {

// d/dz k_0(z) = d/dz [(pi/2z) * e^(-z)]
//             = (pi/2) * d/dz [e^(-z) / z]
//             = (pi/2) * [(-e^(-z) * z - e^(-z)) / z^2]
//             = (pi/2) * e^(-z) * [(-z - 1) / z^2]
//             = -(pi/2) * e^(-z) * (1 + z) / z^2
//             = -k_1(z)
template <typename T>
T spherical_bessel_k_0_backward(T grad_output, T z) {
    const T pi_over_2 = T(1.5707963267948966192313216916398);
    const T eps = detail::spherical_bessel_k_0_eps<T>();

    // For small z, use series expansion
    // d/dz k_0(z) = -(pi/2) * (1+z) * e^(-z) / z^2
    //            ≈ -(pi/2) * (1+z) * (1-z+z^2/2-...) / z^2
    //            ≈ -(pi/2) / z^2 for very small z (dominant term)
    if (std::abs(z) < eps) {
        T z2 = z * z;
        T deriv = -pi_over_2 / z2;
        return grad_output * deriv;
    }

    T exp_neg_z = std::exp(-z);
    T z2 = z * z;
    T deriv = -pi_over_2 * exp_neg_z * (T(1) + z) / z2;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    const T pi_over_2 = T(1.5707963267948966192313216916398);
    const T eps = detail::spherical_bessel_k_0_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> deriv = -c10::complex<T>(pi_over_2, T(0)) / z2;
        return grad_output * std::conj(deriv);
    }

    c10::complex<T> exp_neg_z = std::exp(-z);
    c10::complex<T> z2 = z * z;
    c10::complex<T> deriv = -c10::complex<T>(pi_over_2, T(0)) * exp_neg_z
                           * (c10::complex<T>(T(1), T(0)) + z) / z2;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
