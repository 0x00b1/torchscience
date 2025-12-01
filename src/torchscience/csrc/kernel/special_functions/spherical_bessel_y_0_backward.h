#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_y_0.h"

namespace torchscience::kernel::special_functions {

// d/dz y_0(z) = d/dz [-cos(z)/z] = sin(z)/z + cos(z)/z^2 = -y_1(z)
// Direct formula: (sin(z)*z + cos(z)) / z^2
template <typename T>
T spherical_bessel_y_0_backward(T grad_output, T z) {
    // Gradient is undefined at z=0 (singular point)
    if (z == T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    T sin_z = std::sin(z);
    T cos_z = std::cos(z);
    T z2 = z * z;
    T deriv = (sin_z * z + cos_z) / z2;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y_0_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    // Gradient is undefined at z=0 (singular point)
    if (z == c10::complex<T>(T(0), T(0))) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN());
    }

    c10::complex<T> sin_z = std::sin(z);
    c10::complex<T> cos_z = std::cos(z);
    c10::complex<T> z2 = z * z;
    c10::complex<T> deriv = (sin_z * z + cos_z) / z2;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
