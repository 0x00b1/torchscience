#pragma once

#include <c10/util/complex.h>
#include <cmath>
#include <limits>
#include <tuple>

#include "theta_3.h"

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
inline T theta3_finite_diff_step() {
    return std::pow(std::numeric_limits<T>::epsilon(), T(1.0/3.0));
}

template <>
inline float theta3_finite_diff_step<float>() {
    return 1e-3f;
}

template <>
inline double theta3_finite_diff_step<double>() {
    return 1e-6;
}

} // namespace detail

template <typename T>
std::tuple<T, T> theta_3_backward(
    T gradient,
    T z,
    T q
) {
    T h = detail::theta3_finite_diff_step<T>();

    T f_z_plus = theta_3(z + h, q);
    T f_z_minus = theta_3(z - h, q);
    T derivative_z = (f_z_plus - f_z_minus) / (T(2) * h);

    T f_q_plus = theta_3(z, q + h);
    T f_q_minus = theta_3(z, q - h);
    T derivative_q = (f_q_plus - f_q_minus) / (T(2) * h);

    return {gradient * derivative_z, gradient * derivative_q};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> theta_3_backward(
    c10::complex<T> gradient,
    c10::complex<T> z,
    c10::complex<T> q
) {
    T h = detail::theta3_finite_diff_step<T>();
    c10::complex<T> h_complex(h, T(0));

    c10::complex<T> f_z_plus = theta_3(z + h_complex, q);
    c10::complex<T> f_z_minus = theta_3(z - h_complex, q);
    c10::complex<T> derivative_z = (f_z_plus - f_z_minus) / (T(2) * h_complex);

    c10::complex<T> f_q_plus = theta_3(z, q + h_complex);
    c10::complex<T> f_q_minus = theta_3(z, q - h_complex);
    c10::complex<T> derivative_q = (f_q_plus - f_q_minus) / (T(2) * h_complex);

    return {
        gradient * std::conj(derivative_z),
        gradient * std::conj(derivative_q)
    };
}

} // namespace torchscience::kernel::special_functions
