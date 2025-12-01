#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_j_0.h"
#include "spherical_bessel_j_1.h"

namespace torchscience::kernel::special_functions {

// d/dz j_1(z) = j_0(z) - 2*j_1(z)/z
// For small z, Taylor series: j_1'(z) = 1/3 - z^2/10 + z^4/168 - ...

template <typename T>
T spherical_bessel_j_1_backward(T grad_output, T z) {
    const T eps = detail::spherical_bessel_j_1_eps<T>();

    if (std::abs(z) < eps) {
        T z2 = z * z;
        T deriv = T(1) / T(3) - z2 / T(10) + z2 * z2 / T(168);
        return grad_output * deriv;
    }

    T j0 = spherical_bessel_j_0(z);
    T j1 = spherical_bessel_j_1(z);
    T deriv = j0 - T(2) * j1 / z;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_j_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_j_1_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> deriv = c10::complex<T>(T(1) / T(3), T(0))
                               - z2 / c10::complex<T>(T(10), T(0))
                               + z2 * z2 / c10::complex<T>(T(168), T(0));
        return grad_output * std::conj(deriv);
    }

    c10::complex<T> j0 = spherical_bessel_j_0(z);
    c10::complex<T> j1 = spherical_bessel_j_1(z);
    c10::complex<T> deriv = j0 - c10::complex<T>(T(2), T(0)) * j1 / z;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
