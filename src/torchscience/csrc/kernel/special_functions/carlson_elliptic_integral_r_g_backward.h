#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_g.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_g_backward(
    T gradient,
    T x,
    T y,
    T z
) {
    // Compute gradients using finite differences for robustness
    // The analytical formulas for R_G gradients can be derived from the
    // R_F and R_D gradients, but finite differences are more robust.
    //
    // Note: R_G(x,y,z) = (1/2)[z*R_F(x,y,z) - (1/3)(x-z)(y-z)*R_D(x,y,z) + sqrt(xy/z)]
    //
    // The partial derivatives are complex due to the reordering for numerical stability.

    T eps = std::cbrt(std::numeric_limits<T>::epsilon());

    // Use central differences for better accuracy
    T rg_x_plus = carlson_elliptic_integral_r_g(x + eps, y, z);
    T rg_x_minus = carlson_elliptic_integral_r_g(x - eps, y, z);
    T dx = (rg_x_plus - rg_x_minus) / (T(2) * eps);

    T rg_y_plus = carlson_elliptic_integral_r_g(x, y + eps, z);
    T rg_y_minus = carlson_elliptic_integral_r_g(x, y - eps, z);
    T dy = (rg_y_plus - rg_y_minus) / (T(2) * eps);

    T rg_z_plus = carlson_elliptic_integral_r_g(x, y, z + eps);
    T rg_z_minus = carlson_elliptic_integral_r_g(x, y, z - eps);
    T dz = (rg_z_plus - rg_z_minus) / (T(2) * eps);

    return {gradient * dx, gradient * dy, gradient * dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_g_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    T eps = std::cbrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> eps_c(eps, T(0));

    // Use central differences for better accuracy
    c10::complex<T> rg_x_plus = carlson_elliptic_integral_r_g(x + eps_c, y, z);
    c10::complex<T> rg_x_minus = carlson_elliptic_integral_r_g(x - eps_c, y, z);
    c10::complex<T> dx = (rg_x_plus - rg_x_minus) / (T(2) * eps);

    c10::complex<T> rg_y_plus = carlson_elliptic_integral_r_g(x, y + eps_c, z);
    c10::complex<T> rg_y_minus = carlson_elliptic_integral_r_g(x, y - eps_c, z);
    c10::complex<T> dy = (rg_y_plus - rg_y_minus) / (T(2) * eps);

    c10::complex<T> rg_z_plus = carlson_elliptic_integral_r_g(x, y, z + eps_c);
    c10::complex<T> rg_z_minus = carlson_elliptic_integral_r_g(x, y, z - eps_c);
    c10::complex<T> dz = (rg_z_plus - rg_z_minus) / (T(2) * eps);

    return {gradient * std::conj(dx), gradient * std::conj(dy), gradient * std::conj(dz)};
}

} // namespace torchscience::kernel::special_functions
