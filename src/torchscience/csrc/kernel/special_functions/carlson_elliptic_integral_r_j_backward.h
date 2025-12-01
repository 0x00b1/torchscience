#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_j.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> carlson_elliptic_integral_r_j_backward(
    T gradient,
    T x,
    T y,
    T z,
    T p
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for R_J gradients are complex and involve
    // careful handling of near-singular cases

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    T rj_x_plus = carlson_elliptic_integral_r_j(x + eps, y, z, p);
    T rj_x_minus = carlson_elliptic_integral_r_j(x - eps, y, z, p);
    T dx = (rj_x_plus - rj_x_minus) / (T(2) * eps);

    T rj_y_plus = carlson_elliptic_integral_r_j(x, y + eps, z, p);
    T rj_y_minus = carlson_elliptic_integral_r_j(x, y - eps, z, p);
    T dy = (rj_y_plus - rj_y_minus) / (T(2) * eps);

    T rj_z_plus = carlson_elliptic_integral_r_j(x, y, z + eps, p);
    T rj_z_minus = carlson_elliptic_integral_r_j(x, y, z - eps, p);
    T dz = (rj_z_plus - rj_z_minus) / (T(2) * eps);

    T rj_p_plus = carlson_elliptic_integral_r_j(x, y, z, p + eps);
    T rj_p_minus = carlson_elliptic_integral_r_j(x, y, z, p - eps);
    T dp = (rj_p_plus - rj_p_minus) / (T(2) * eps);

    return {gradient * dx, gradient * dy, gradient * dz, gradient * dp};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_j_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z,
    c10::complex<T> p
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    c10::complex<T> rj_x_plus = carlson_elliptic_integral_r_j(x + eps_c, y, z, p);
    c10::complex<T> rj_x_minus = carlson_elliptic_integral_r_j(x - eps_c, y, z, p);
    c10::complex<T> dx = (rj_x_plus - rj_x_minus) / (T(2) * eps);

    c10::complex<T> rj_y_plus = carlson_elliptic_integral_r_j(x, y + eps_c, z, p);
    c10::complex<T> rj_y_minus = carlson_elliptic_integral_r_j(x, y - eps_c, z, p);
    c10::complex<T> dy = (rj_y_plus - rj_y_minus) / (T(2) * eps);

    c10::complex<T> rj_z_plus = carlson_elliptic_integral_r_j(x, y, z + eps_c, p);
    c10::complex<T> rj_z_minus = carlson_elliptic_integral_r_j(x, y, z - eps_c, p);
    c10::complex<T> dz = (rj_z_plus - rj_z_minus) / (T(2) * eps);

    c10::complex<T> rj_p_plus = carlson_elliptic_integral_r_j(x, y, z, p + eps_c);
    c10::complex<T> rj_p_minus = carlson_elliptic_integral_r_j(x, y, z, p - eps_c);
    c10::complex<T> dp = (rj_p_plus - rj_p_minus) / (T(2) * eps);

    return {gradient * std::conj(dx), gradient * std::conj(dy),
            gradient * std::conj(dz), gradient * std::conj(dp)};
}

} // namespace torchscience::kernel::special_functions
