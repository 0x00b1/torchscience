#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_i_0.h"
#include "spherical_bessel_i_0_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> spherical_bessel_i_0_backward_backward(T gg_z, T grad_output, T z) {
    const T eps = detail::spherical_bessel_i_0_eps<T>();

    T first_deriv;
    T second_deriv;

    if (std::abs(z) < eps) {
        T z2 = z * z;
        first_deriv = z / T(3) + z * z2 / T(30);
        second_deriv = T(1) / T(3) + z2 / T(10);
    } else {
        T sinh_z = std::sinh(z);
        T cosh_z = std::cosh(z);
        T z2 = z * z;
        T z3 = z2 * z;

        first_deriv = (cosh_z * z - sinh_z) / z2;
        // d^2/dz^2 i_0(z) = sinh(z)/z - 2*cosh(z)/z^2 + 2*sinh(z)/z^3
        second_deriv = sinh_z / z - T(2) * cosh_z / z2 + T(2) * sinh_z / z3;
    }

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_i_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_i_0_eps<T>();

    c10::complex<T> first_deriv;
    c10::complex<T> second_deriv;

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        first_deriv = z / c10::complex<T>(T(3), T(0))
                     + z * z2 / c10::complex<T>(T(30), T(0));
        second_deriv = c10::complex<T>(T(1) / T(3), T(0))
                      + z2 / c10::complex<T>(T(10), T(0));
    } else {
        c10::complex<T> sinh_z = std::sinh(z);
        c10::complex<T> cosh_z = std::cosh(z);
        c10::complex<T> z2 = z * z;
        c10::complex<T> z3 = z2 * z;

        first_deriv = (cosh_z * z - sinh_z) / z2;
        second_deriv = sinh_z / z - c10::complex<T>(T(2), T(0)) * cosh_z / z2
                      + c10::complex<T>(T(2), T(0)) * sinh_z / z3;
    }

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
