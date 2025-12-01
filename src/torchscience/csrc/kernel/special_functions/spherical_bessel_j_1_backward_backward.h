#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_j_0.h"
#include "spherical_bessel_j_1.h"

namespace torchscience::kernel::special_functions {

// j_1''(z) = -j_1(z) - 2*j_0(z)/z + 6*j_1(z)/z^2
// For small z, Taylor series: j_1''(z) = -z/5 + z^3/42 - ...

template <typename T>
std::tuple<T, T> spherical_bessel_j_1_backward_backward(T gg_z, T grad_output, T z) {
    const T eps = detail::spherical_bessel_j_1_eps<T>();

    T first_deriv;
    T second_deriv;

    if (std::abs(z) < eps) {
        T z2 = z * z;
        first_deriv = T(1) / T(3) - z2 / T(10) + z2 * z2 / T(168);
        second_deriv = -z / T(5) + z * z2 / T(42);
    } else {
        T j0 = spherical_bessel_j_0(z);
        T j1 = spherical_bessel_j_1(z);
        T z2 = z * z;

        first_deriv = j0 - T(2) * j1 / z;
        second_deriv = -j1 - T(2) * j0 / z + T(6) * j1 / z2;
    }

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_j_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_j_1_eps<T>();

    c10::complex<T> first_deriv;
    c10::complex<T> second_deriv;

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        first_deriv = c10::complex<T>(T(1) / T(3), T(0))
                     - z2 / c10::complex<T>(T(10), T(0))
                     + z2 * z2 / c10::complex<T>(T(168), T(0));
        second_deriv = -z / c10::complex<T>(T(5), T(0))
                      + z * z2 / c10::complex<T>(T(42), T(0));
    } else {
        c10::complex<T> j0 = spherical_bessel_j_0(z);
        c10::complex<T> j1 = spherical_bessel_j_1(z);
        c10::complex<T> z2 = z * z;

        first_deriv = j0 - c10::complex<T>(T(2), T(0)) * j1 / z;
        second_deriv = -j1 - c10::complex<T>(T(2), T(0)) * j0 / z
                      + c10::complex<T>(T(6), T(0)) * j1 / z2;
    }

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
