#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_j_0.h"

namespace torchscience::kernel::special_functions {

// d/dz j_0(z) = d/dz [sin(z)/z] = cos(z)/z - sin(z)/z^2 = -j_1(z)
// Direct formula: (cos(z)*z - sin(z)) / z^2
template <typename T>
T spherical_bessel_j_0_backward(T grad_output, T z) {
    const T eps = detail::spherical_bessel_j_0_eps<T>();

    // For small z, use Taylor series: j_0'(z) = -z/3 + z^3/30 - ...
    if (std::abs(z) < eps) {
        T z2 = z * z;
        T deriv = -z / T(3) + z * z2 / T(30);
        return grad_output * deriv;
    }

    T sin_z = std::sin(z);
    T cos_z = std::cos(z);
    T z2 = z * z;
    T deriv = (cos_z * z - sin_z) / z2;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_j_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_j_0_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> deriv = -z / c10::complex<T>(T(3), T(0))
                                + z * z2 / c10::complex<T>(T(30), T(0));
        return grad_output * std::conj(deriv);
    }

    c10::complex<T> sin_z = std::sin(z);
    c10::complex<T> cos_z = std::cos(z);
    c10::complex<T> z2 = z * z;
    c10::complex<T> deriv = (cos_z * z - sin_z) / z2;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
