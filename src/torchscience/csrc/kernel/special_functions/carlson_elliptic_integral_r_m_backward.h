#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_m.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_m_backward(
    T gradient,
    T x,
    T y,
    T z
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for R_M gradients are complex due to
    // the relationship with R_F and R_J.

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    T rm_x_plus = carlson_elliptic_integral_r_m(x + eps, y, z);
    T rm_x_minus = carlson_elliptic_integral_r_m(x - eps, y, z);
    T dx = (rm_x_plus - rm_x_minus) / (T(2) * eps);

    T rm_y_plus = carlson_elliptic_integral_r_m(x, y + eps, z);
    T rm_y_minus = carlson_elliptic_integral_r_m(x, y - eps, z);
    T dy = (rm_y_plus - rm_y_minus) / (T(2) * eps);

    T rm_z_plus = carlson_elliptic_integral_r_m(x, y, z + eps);
    T rm_z_minus = carlson_elliptic_integral_r_m(x, y, z - eps);
    T dz = (rm_z_plus - rm_z_minus) / (T(2) * eps);

    return {gradient * dx, gradient * dy, gradient * dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_m_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    c10::complex<T> rm_x_plus = carlson_elliptic_integral_r_m(x + eps_c, y, z);
    c10::complex<T> rm_x_minus = carlson_elliptic_integral_r_m(x - eps_c, y, z);
    c10::complex<T> dx = (rm_x_plus - rm_x_minus) / (T(2) * eps);

    c10::complex<T> rm_y_plus = carlson_elliptic_integral_r_m(x, y + eps_c, z);
    c10::complex<T> rm_y_minus = carlson_elliptic_integral_r_m(x, y - eps_c, z);
    c10::complex<T> dy = (rm_y_plus - rm_y_minus) / (T(2) * eps);

    c10::complex<T> rm_z_plus = carlson_elliptic_integral_r_m(x, y, z + eps_c);
    c10::complex<T> rm_z_minus = carlson_elliptic_integral_r_m(x, y, z - eps_c);
    c10::complex<T> dz = (rm_z_plus - rm_z_minus) / (T(2) * eps);

    return {gradient * std::conj(dx), gradient * std::conj(dy), gradient * std::conj(dz)};
}

} // namespace torchscience::kernel::special_functions
