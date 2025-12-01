#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_k_0.h"
#include "spherical_bessel_k_0_backward.h"

namespace torchscience::kernel::special_functions {

// First derivative: k_0'(z) = -(pi/2) * e^(-z) * (1+z) / z^2
// Second derivative: k_0''(z) = d/dz [-(pi/2) * e^(-z) * (1+z) / z^2]
//
// Let g(z) = e^(-z) * (1+z) / z^2
// g'(z) = d/dz [e^(-z) * (1+z)] / z^2 + e^(-z) * (1+z) * d/dz[1/z^2]
//       = [e^(-z) - e^(-z) * (1+z)] / z^2 + e^(-z) * (1+z) * (-2/z^3)
//       = e^(-z) * [1 - (1+z)] / z^2 - 2 * e^(-z) * (1+z) / z^3
//       = -e^(-z) * z / z^2 - 2 * e^(-z) * (1+z) / z^3
//       = -e^(-z) / z - 2 * e^(-z) * (1+z) / z^3
//       = e^(-z) * [-z^2 - 2(1+z)] / z^3
//       = e^(-z) * [-z^2 - 2 - 2z] / z^3
//       = -e^(-z) * (z^2 + 2z + 2) / z^3
//
// So k_0''(z) = -(pi/2) * g'(z) = (pi/2) * e^(-z) * (z^2 + 2z + 2) / z^3
template <typename T>
std::tuple<T, T> spherical_bessel_k_0_backward_backward(T gg_z, T grad_output, T z) {
    const T pi_over_2 = T(1.5707963267948966192313216916398);
    const T eps = detail::spherical_bessel_k_0_eps<T>();

    T first_deriv;
    T second_deriv;

    if (std::abs(z) < eps) {
        T z2 = z * z;
        T z3 = z2 * z;
        // Leading terms for small z:
        // first_deriv ≈ -(pi/2) / z^2
        // second_deriv ≈ (pi/2) * 2 / z^3
        first_deriv = -pi_over_2 / z2;
        second_deriv = pi_over_2 * T(2) / z3;
    } else {
        T exp_neg_z = std::exp(-z);
        T z2 = z * z;
        T z3 = z2 * z;

        first_deriv = -pi_over_2 * exp_neg_z * (T(1) + z) / z2;
        second_deriv = pi_over_2 * exp_neg_z * (z2 + T(2) * z + T(2)) / z3;
    }

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_k_0_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    const T pi_over_2 = T(1.5707963267948966192313216916398);
    const T eps = detail::spherical_bessel_k_0_eps<T>();

    c10::complex<T> first_deriv;
    c10::complex<T> second_deriv;

    if (std::abs(z) < eps) {
        c10::complex<T> z2 = z * z;
        c10::complex<T> z3 = z2 * z;
        first_deriv = -c10::complex<T>(pi_over_2, T(0)) / z2;
        second_deriv = c10::complex<T>(pi_over_2 * T(2), T(0)) / z3;
    } else {
        c10::complex<T> exp_neg_z = std::exp(-z);
        c10::complex<T> z2 = z * z;
        c10::complex<T> z3 = z2 * z;

        first_deriv = -c10::complex<T>(pi_over_2, T(0)) * exp_neg_z
                     * (c10::complex<T>(T(1), T(0)) + z) / z2;
        second_deriv = c10::complex<T>(pi_over_2, T(0)) * exp_neg_z
                      * (z2 + c10::complex<T>(T(2), T(0)) * z + c10::complex<T>(T(2), T(0))) / z3;
    }

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
