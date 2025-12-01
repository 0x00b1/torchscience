#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_y_0.h"
#include "spherical_bessel_y_0_backward.h"

namespace torchscience::kernel::special_functions {

// Second derivative of y_0(z):
// d²/dz² y_0(z) = -y_0(z) - (2/z) * y_0'(z)
// where y_0(z) = -cos(z)/z and y_0'(z) = (sin(z)*z + cos(z))/z²
//
// Expanding: d²/dz² y_0(z) = cos(z)/z - 2*sin(z)/z² - 2*cos(z)/z³
template <typename T>
std::tuple<T, T> spherical_bessel_y_0_backward_backward(T gg_z, T grad_output, T z) {
    // Undefined at z=0
    if (z == T(0)) {
        return {std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()};
    }

    T sin_z = std::sin(z);
    T cos_z = std::cos(z);
    T z2 = z * z;
    T z3 = z2 * z;

    T first_deriv = (sin_z * z + cos_z) / z2;
    T second_deriv = cos_z / z - T(2) * sin_z / z2 - T(2) * cos_z / z3;

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_y_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    // Undefined at z=0
    if (z == c10::complex<T>(T(0), T(0))) {
        return {c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN()),
                c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::quiet_NaN())};
    }

    c10::complex<T> sin_z = std::sin(z);
    c10::complex<T> cos_z = std::cos(z);
    c10::complex<T> z2 = z * z;
    c10::complex<T> z3 = z2 * z;

    c10::complex<T> first_deriv = (sin_z * z + cos_z) / z2;
    c10::complex<T> second_deriv = cos_z / z - c10::complex<T>(T(2), T(0)) * sin_z / z2
                                  - c10::complex<T>(T(2), T(0)) * cos_z / z3;

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
