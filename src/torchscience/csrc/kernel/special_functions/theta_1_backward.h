#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "theta_1.h"

namespace torchscience::kernel::special_functions {

// Backward pass for Jacobi theta function theta_1(z, q)
// Uses numerical finite differences for derivatives.

namespace detail {

template <typename T>
inline T theta1_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float theta1_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double theta1_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> theta_1_backward(
    T gradient,
    T z,
    T q
) {
    T h = detail::theta1_finite_diff_step<T>();

    // Gradient w.r.t. z
    T f_z_plus = theta_1(z + h, q);
    T f_z_minus = theta_1(z - h, q);
    T derivative_z = (f_z_plus - f_z_minus) / (T(2) * h);

    // Gradient w.r.t. q
    T f_q_plus = theta_1(z, q + h);
    T f_q_minus = theta_1(z, q - h);
    T derivative_q = (f_q_plus - f_q_minus) / (T(2) * h);

    return {gradient * derivative_z, gradient * derivative_q};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> theta_1_backward(
    c10::complex<T> gradient,
    c10::complex<T> z,
    c10::complex<T> q
) {
    T h = detail::theta1_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    // Gradient w.r.t. z
    c10::complex<T> f_z_plus = theta_1(z + h_complex, q);
    c10::complex<T> f_z_minus = theta_1(z - h_complex, q);
    c10::complex<T> derivative_z = (f_z_plus - f_z_minus) / (T(2) * h_complex);

    // Gradient w.r.t. q
    c10::complex<T> f_q_plus = theta_1(z, q + h_complex);
    c10::complex<T> f_q_minus = theta_1(z, q - h_complex);
    c10::complex<T> derivative_q = (f_q_plus - f_q_minus) / (T(2) * h_complex);

    return {
        gradient * std::conj(derivative_z),
        gradient * std::conj(derivative_q)
    };
}

} // namespace torchscience::kernel::special_functions
