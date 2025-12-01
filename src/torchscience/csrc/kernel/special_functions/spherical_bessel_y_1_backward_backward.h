#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_y_0.h"
#include "spherical_bessel_y_1.h"

namespace torchscience::kernel::special_functions {

// y_1''(z) = -y_1(z) - 2*y_0(z)/z + 6*y_1(z)/z^2
// Note: Both y_0 and y_1 are singular at z=0, so no Taylor series is used

template <typename T>
std::tuple<T, T> spherical_bessel_y_1_backward_backward(T gg_z, T grad_output, T z) {
    T y0 = spherical_bessel_y_0(z);
    T y1 = spherical_bessel_y_1(z);
    T z2 = z * z;

    T first_deriv = y0 - T(2) * y1 / z;
    T second_deriv = -y1 - T(2) * y0 / z + T(6) * y1 / z2;

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_y_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    c10::complex<T> y0 = spherical_bessel_y_0(z);
    c10::complex<T> y1 = spherical_bessel_y_1(z);
    c10::complex<T> z2 = z * z;

    c10::complex<T> first_deriv = y0 - c10::complex<T>(T(2), T(0)) * y1 / z;
    c10::complex<T> second_deriv = -y1 - c10::complex<T>(T(2), T(0)) * y0 / z
                  + c10::complex<T>(T(6), T(0)) * y1 / z2;

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
