#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_d.h"
#include "carlson_elliptic_integral_r_d_backward.h"
#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_f_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> carlson_elliptic_integral_r_d_backward_backward(
    T gg_x,
    T gg_y,
    T gg_z,
    T gradient,
    T x,
    T y,
    T z
) {
    // Numerical approximation for second derivatives
    // This is a placeholder - full analytical derivatives are complex
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dx, dy, dz] = carlson_elliptic_integral_r_d_backward(T(1), x, y, z);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz;

    // Second derivatives via finite differences (placeholder)
    auto [dx_px, dy_px, dz_px] = carlson_elliptic_integral_r_d_backward(T(1), x + eps, y, z);
    auto [dx_py, dy_py, dz_py] = carlson_elliptic_integral_r_d_backward(T(1), x, y + eps, z);
    auto [dx_pz, dy_pz, dz_pz] = carlson_elliptic_integral_r_d_backward(T(1), x, y, z + eps);

    T d2xx = (dx_px - dx) / eps;
    T d2xy = (dx_py - dx) / eps;
    T d2xz = (dx_pz - dx) / eps;
    T d2yx = (dy_px - dy) / eps;
    T d2yy = (dy_py - dy) / eps;
    T d2yz = (dy_pz - dy) / eps;
    T d2zx = (dz_px - dz) / eps;
    T d2zy = (dz_py - dz) / eps;
    T d2zz = (dz_pz - dz) / eps;

    T grad_x = gradient * (gg_x * d2xx + gg_y * d2yx + gg_z * d2zx);
    T grad_y = gradient * (gg_x * d2xy + gg_y * d2yy + gg_z * d2zy);
    T grad_z = gradient * (gg_x * d2xz + gg_y * d2yz + gg_z * d2zz);

    return {grad_gradient, grad_x, grad_y, grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_d_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gg_z,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dx, dy, dz] = carlson_elliptic_integral_r_d_backward(c10::complex<T>(T(1), T(0)), x, y, z);

    c10::complex<T> grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz;

    auto [dx_px, dy_px, dz_px] = carlson_elliptic_integral_r_d_backward(
        c10::complex<T>(T(1), T(0)), x + c10::complex<T>(eps, T(0)), y, z);
    auto [dx_py, dy_py, dz_py] = carlson_elliptic_integral_r_d_backward(
        c10::complex<T>(T(1), T(0)), x, y + c10::complex<T>(eps, T(0)), z);
    auto [dx_pz, dy_pz, dz_pz] = carlson_elliptic_integral_r_d_backward(
        c10::complex<T>(T(1), T(0)), x, y, z + c10::complex<T>(eps, T(0)));

    c10::complex<T> d2xx = (dx_px - dx) / eps;
    c10::complex<T> d2xy = (dx_py - dx) / eps;
    c10::complex<T> d2xz = (dx_pz - dx) / eps;
    c10::complex<T> d2yx = (dy_px - dy) / eps;
    c10::complex<T> d2yy = (dy_py - dy) / eps;
    c10::complex<T> d2yz = (dy_pz - dy) / eps;
    c10::complex<T> d2zx = (dz_px - dz) / eps;
    c10::complex<T> d2zy = (dz_py - dz) / eps;
    c10::complex<T> d2zz = (dz_pz - dz) / eps;

    c10::complex<T> grad_x = gradient * (gg_x * d2xx + gg_y * d2yx + gg_z * d2zx);
    c10::complex<T> grad_y = gradient * (gg_x * d2xy + gg_y * d2yy + gg_z * d2zy);
    c10::complex<T> grad_z = gradient * (gg_x * d2xz + gg_y * d2yz + gg_z * d2zz);

    return {grad_gradient, grad_x, grad_y, grad_z};
}

} // namespace torchscience::kernel::special_functions
