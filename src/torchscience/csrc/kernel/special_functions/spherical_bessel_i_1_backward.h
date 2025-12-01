#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "spherical_bessel_i_0.h"
#include "spherical_bessel_i_1.h"

namespace torchscience::kernel::special_functions {

// d/dz i_1(z) = i_0(z) - 2*i_1(z)/z (note: + sign for i_0, same pattern as j_1 but different function)
// For small z, Taylor series: i_1'(z) = 1/3 + z^2/10 + z^4/168 + ...

template <typename T>
T spherical_bessel_i_1_backward(T grad_output, T z) {
    const T eps = detail::spherical_bessel_i_1_eps<T>();

    if (std::abs(z) < eps) {
        T z2 = z * z;
        T deriv = T(1) / T(3) + z2 / T(10) + z2 * z2 / T(168);
        return grad_output * deriv;
    }

    T i0 = spherical_bessel_i_0(z);
    T i1 = spherical_bessel_i_1(z);
    T deriv = i0 - T(2) * i1 / z;

    return grad_output * deriv;
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_i_1_backward(c10::complex<T> grad_output, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_i_1_eps<T>();

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> deriv = c10::complex<T>(T(1) / T(3), T(0))
                               + z2 / c10::complex<T>(T(10), T(0))
                               + z2 * z2 / c10::complex<T>(T(168), T(0));
        return grad_output * std::conj(deriv);
    }

    c10::complex<T> i0 = spherical_bessel_i_0(z);
    c10::complex<T> i1 = spherical_bessel_i_1(z);
    c10::complex<T> deriv = i0 - c10::complex<T>(T(2), T(0)) * i1 / z;

    return grad_output * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
