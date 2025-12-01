#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_d.h"
#include "carlson_elliptic_integral_r_d_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> carlson_elliptic_integral_r_f_backward_backward(
    T gg_x,
    T gg_y,
    T gg_z,
    T gradient,
    T x,
    T y,
    T z
) {
    // First derivatives: df/dx = -R_D(y,z,x)/6, etc.
    T df_dx = -carlson_elliptic_integral_r_d(y, z, x) / T(6);
    T df_dy = -carlson_elliptic_integral_r_d(z, x, y) / T(6);
    T df_dz = -carlson_elliptic_integral_r_d(x, y, z) / T(6);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_x * df_dx + gg_y * df_dy + gg_z * df_dz;

    // Second derivatives via R_D backward
    auto [d2_xx_y, d2_xx_z, d2_xx_x] = carlson_elliptic_integral_r_d_backward(T(1), y, z, x);
    auto [d2_yy_z, d2_yy_x, d2_yy_y] = carlson_elliptic_integral_r_d_backward(T(1), z, x, y);
    auto [d2_zz_x, d2_zz_y, d2_zz_z] = carlson_elliptic_integral_r_d_backward(T(1), x, y, z);

    T grad_x = gradient * (gg_x * (-d2_xx_x / T(6)) + gg_y * (-d2_yy_x / T(6)) + gg_z * (-d2_zz_x / T(6)));
    T grad_y = gradient * (gg_x * (-d2_xx_y / T(6)) + gg_y * (-d2_yy_y / T(6)) + gg_z * (-d2_zz_y / T(6)));
    T grad_z = gradient * (gg_x * (-d2_xx_z / T(6)) + gg_y * (-d2_yy_z / T(6)) + gg_z * (-d2_zz_z / T(6)));

    return {grad_gradient, grad_x, grad_y, grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_f_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gg_z,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    c10::complex<T> df_dx = -carlson_elliptic_integral_r_d(y, z, x) / T(6);
    c10::complex<T> df_dy = -carlson_elliptic_integral_r_d(z, x, y) / T(6);
    c10::complex<T> df_dz = -carlson_elliptic_integral_r_d(x, y, z) / T(6);

    c10::complex<T> grad_gradient = gg_x * std::conj(df_dx) + gg_y * std::conj(df_dy) + gg_z * std::conj(df_dz);

    auto [d2_xx_y, d2_xx_z, d2_xx_x] = carlson_elliptic_integral_r_d_backward(c10::complex<T>(T(1), T(0)), y, z, x);
    auto [d2_yy_z, d2_yy_x, d2_yy_y] = carlson_elliptic_integral_r_d_backward(c10::complex<T>(T(1), T(0)), z, x, y);
    auto [d2_zz_x, d2_zz_y, d2_zz_z] = carlson_elliptic_integral_r_d_backward(c10::complex<T>(T(1), T(0)), x, y, z);

    // Note: backward already applies conjugation, so we need to undo it for the chain rule
    c10::complex<T> grad_x = gradient * (gg_x * (-d2_xx_x / T(6)) + gg_y * (-d2_yy_x / T(6)) + gg_z * (-d2_zz_x / T(6)));
    c10::complex<T> grad_y = gradient * (gg_x * (-d2_xx_y / T(6)) + gg_y * (-d2_yy_y / T(6)) + gg_z * (-d2_zz_y / T(6)));
    c10::complex<T> grad_z = gradient * (gg_x * (-d2_xx_z / T(6)) + gg_y * (-d2_yy_z / T(6)) + gg_z * (-d2_zz_z / T(6)));

    return {grad_gradient, grad_x, grad_y, grad_z};
}

} // namespace torchscience::kernel::special_functions
