#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "spherical_bessel_k_1.h"

namespace torchscience::kernel::special_functions {

// Second derivative of k_1(z):
// d^2/dz^2 k_1(z) = (pi/2z^4)(6 + 6z + 3z^2 + z^3) e^(-z)
//
// Derivation from d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z):
// Let f(z) = -(pi/2) * (2 + 2z + z^2) * z^(-3) * e^(-z)
// Using product rule for uvw:
// u = (2 + 2z + z^2), u' = (2 + 2z)
// v = z^(-3), v' = -3z^(-4)
// w = e^(-z), w' = -e^(-z)
//
// f'(z) = -(pi/2) * e^(-z) * z^(-4) * [(2 + 2z)z - 3(2 + 2z + z^2) - (2 + 2z + z^2)z]
//       = -(pi/2) * e^(-z) * z^(-4) * [2z + 2z^2 - 6 - 6z - 3z^2 - 2z - 2z^2 - z^3]
//       = -(pi/2) * e^(-z) * z^(-4) * [-6 - 6z - 3z^2 - z^3]
//       = (pi/2) * e^(-z) * z^(-4) * (6 + 6z + 3z^2 + z^3)

template <typename T>
std::tuple<T, T> spherical_bessel_k_1_backward_backward(T gg_z, T grad_output, T z) {
    const T pi = T(3.14159265358979323846);
    const T eps = detail::spherical_bessel_k_1_eps<T>();

    T first_deriv;
    T second_deriv;

    if (std::abs(z) < eps) {
        // Near pole, derivatives are unbounded
        first_deriv = -std::numeric_limits<T>::infinity();
        second_deriv = std::numeric_limits<T>::infinity();
    } else {
        T z_inv = T(1) / z;
        T z2 = z * z;
        T z3 = z2 * z;
        T z3_inv = z_inv * z_inv * z_inv;
        T z4_inv = z3_inv * z_inv;
        T exp_neg_z = std::exp(-z);

        // d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z)
        first_deriv = -(pi / T(2)) * z3_inv * (T(2) + T(2) * z + z2) * exp_neg_z;

        // d^2/dz^2 k_1(z) = (pi/2z^4)(6 + 6z + 3z^2 + z^3) e^(-z)
        second_deriv = (pi / T(2)) * z4_inv * (T(6) + T(6) * z + T(3) * z2 + z3) * exp_neg_z;
    }

    T grad_grad_output = gg_z * first_deriv;
    T grad_z = gg_z * grad_output * second_deriv;

    return {grad_grad_output, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> spherical_bessel_k_1_backward_backward(
    c10::complex<T> gg_z, c10::complex<T> grad_output, c10::complex<T> z) {
    const T pi = T(3.14159265358979323846);
    const T eps = detail::spherical_bessel_k_1_eps<T>();

    c10::complex<T> first_deriv;
    c10::complex<T> second_deriv;

    if (std::abs(z) < eps) {
        first_deriv = c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
        second_deriv = c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    } else {
        c10::complex<T> z_inv = c10::complex<T>(T(1), T(0)) / z;
        c10::complex<T> z2 = z * z;
        c10::complex<T> z3 = z2 * z;
        c10::complex<T> z3_inv = z_inv * z_inv * z_inv;
        c10::complex<T> z4_inv = z3_inv * z_inv;
        c10::complex<T> exp_neg_z = std::exp(-z);

        c10::complex<T> pi_over_2(pi / T(2), T(0));
        c10::complex<T> two(T(2), T(0));
        c10::complex<T> three(T(3), T(0));
        c10::complex<T> six(T(6), T(0));

        // d/dz k_1(z) = -(pi/2z^3)(2 + 2z + z^2) e^(-z)
        first_deriv = -pi_over_2 * z3_inv * (two + two * z + z2) * exp_neg_z;

        // d^2/dz^2 k_1(z) = (pi/2z^4)(6 + 6z + 3z^2 + z^3) e^(-z)
        second_deriv = pi_over_2 * z4_inv * (six + six * z + three * z2 + z3) * exp_neg_z;
    }

    c10::complex<T> grad_grad_output = gg_z * std::conj(first_deriv);
    c10::complex<T> grad_z = gg_z * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_z};
}

} // namespace torchscience::kernel::special_functions
