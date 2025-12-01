#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_y_0.h"
#include "spherical_bessel_y_1.h"

namespace torchscience::kernel::special_functions {

// d/dz y_1(z) = y_0(z) - 2*y_1(z)/z
// Note: Both y_0 and y_1 are singular at z=0, so no Taylor series is used

template <typename T>
T spherical_bessel_y_1_backward(T grad_output, T z) {
    T y0 = spherical_bessel_y_0(z);
    T y1 = spherical_bessel_y_1(z);
    T deriv = y0 - T(2) * y1 / z;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y0 = spherical_bessel_y_0(z);
    c10::complex<T> y1 = spherical_bessel_y_1(z);
    c10::complex<T> deriv = y0 - c10::complex<T>(T(2), T(0)) * y1 / z;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
