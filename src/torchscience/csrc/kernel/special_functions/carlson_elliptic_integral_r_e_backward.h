#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_e.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_e_backward(
    T gradient,
    T x,
    T y,
    T z
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for R_E gradients are complex due to
    // the relationship with R_D.

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    T re_x_plus = carlson_elliptic_integral_r_e(x + eps, y, z);
    T re_x_minus = carlson_elliptic_integral_r_e(x - eps, y, z);
    T dx = (re_x_plus - re_x_minus) / (T(2) * eps);

    T re_y_plus = carlson_elliptic_integral_r_e(x, y + eps, z);
    T re_y_minus = carlson_elliptic_integral_r_e(x, y - eps, z);
    T dy = (re_y_plus - re_y_minus) / (T(2) * eps);

    T re_z_plus = carlson_elliptic_integral_r_e(x, y, z + eps);
    T re_z_minus = carlson_elliptic_integral_r_e(x, y, z - eps);
    T dz = (re_z_plus - re_z_minus) / (T(2) * eps);

    return {gradient * dx, gradient * dy, gradient * dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_e_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    c10::complex<T> re_x_plus = carlson_elliptic_integral_r_e(x + eps_c, y, z);
    c10::complex<T> re_x_minus = carlson_elliptic_integral_r_e(x - eps_c, y, z);
    c10::complex<T> dx = (re_x_plus - re_x_minus) / (T(2) * eps);

    c10::complex<T> re_y_plus = carlson_elliptic_integral_r_e(x, y + eps_c, z);
    c10::complex<T> re_y_minus = carlson_elliptic_integral_r_e(x, y - eps_c, z);
    c10::complex<T> dy = (re_y_plus - re_y_minus) / (T(2) * eps);

    c10::complex<T> re_z_plus = carlson_elliptic_integral_r_e(x, y, z + eps_c);
    c10::complex<T> re_z_minus = carlson_elliptic_integral_r_e(x, y, z - eps_c);
    c10::complex<T> dz = (re_z_plus - re_z_minus) / (T(2) * eps);

    return {gradient * std::conj(dx), gradient * std::conj(dy), gradient * std::conj(dz)};
}

} // namespace torchscience::kernel::special_functions
