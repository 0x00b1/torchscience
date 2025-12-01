#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_j.h"
#include "carlson_elliptic_integral_r_j_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T, T> carlson_elliptic_integral_r_j_backward_backward(
    T gg_x,
    T gg_y,
    T gg_z,
    T gg_p,
    T gradient,
    T x,
    T y,
    T z,
    T p
) {
    // Numerical approximation for second derivatives
    // This is a placeholder - full analytical derivatives are complex
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dx, dy, dz, dp] = carlson_elliptic_integral_r_j_backward(T(1), x, y, z, p);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz + gg_p * dp;

    // Second derivatives via finite differences (placeholder)
    auto [dx_px, dy_px, dz_px, dp_px] = carlson_elliptic_integral_r_j_backward(T(1), x + eps, y, z, p);
    auto [dx_py, dy_py, dz_py, dp_py] = carlson_elliptic_integral_r_j_backward(T(1), x, y + eps, z, p);
    auto [dx_pz, dy_pz, dz_pz, dp_pz] = carlson_elliptic_integral_r_j_backward(T(1), x, y, z + eps, p);
    auto [dx_pp, dy_pp, dz_pp, dp_pp] = carlson_elliptic_integral_r_j_backward(T(1), x, y, z, p + eps);

    T d2xx = (dx_px - dx) / eps;
    T d2xy = (dx_py - dx) / eps;
    T d2xz = (dx_pz - dx) / eps;
    T d2xp = (dx_pp - dx) / eps;
    T d2yx = (dy_px - dy) / eps;
    T d2yy = (dy_py - dy) / eps;
    T d2yz = (dy_pz - dy) / eps;
    T d2yp = (dy_pp - dy) / eps;
    T d2zx = (dz_px - dz) / eps;
    T d2zy = (dz_py - dz) / eps;
    T d2zz = (dz_pz - dz) / eps;
    T d2zp = (dz_pp - dz) / eps;
    T d2px = (dp_px - dp) / eps;
    T d2py = (dp_py - dp) / eps;
    T d2pz = (dp_pz - dp) / eps;
    T d2pp = (dp_pp - dp) / eps;

    T grad_x = gradient * (gg_x * d2xx + gg_y * d2yx + gg_z * d2zx + gg_p * d2px);
    T grad_y = gradient * (gg_x * d2xy + gg_y * d2yy + gg_z * d2zy + gg_p * d2py);
    T grad_z = gradient * (gg_x * d2xz + gg_y * d2yz + gg_z * d2zz + gg_p * d2pz);
    T grad_p = gradient * (gg_x * d2xp + gg_y * d2yp + gg_z * d2zp + gg_p * d2pp);

    return {grad_gradient, grad_x, grad_y, grad_z, grad_p};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_j_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gg_z,
    c10::complex<T> gg_p,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z,
    c10::complex<T> p
) {
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> eps_c(eps, T(0));

    auto [dx, dy, dz, dp] = carlson_elliptic_integral_r_j_backward(one, x, y, z, p);

    c10::complex<T> grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz + gg_p * dp;

    auto [dx_px, dy_px, dz_px, dp_px] = carlson_elliptic_integral_r_j_backward(one, x + eps_c, y, z, p);
    auto [dx_py, dy_py, dz_py, dp_py] = carlson_elliptic_integral_r_j_backward(one, x, y + eps_c, z, p);
    auto [dx_pz, dy_pz, dz_pz, dp_pz] = carlson_elliptic_integral_r_j_backward(one, x, y, z + eps_c, p);
    auto [dx_pp, dy_pp, dz_pp, dp_pp] = carlson_elliptic_integral_r_j_backward(one, x, y, z, p + eps_c);

    c10::complex<T> d2xx = (dx_px - dx) / eps;
    c10::complex<T> d2xy = (dx_py - dx) / eps;
    c10::complex<T> d2xz = (dx_pz - dx) / eps;
    c10::complex<T> d2xp = (dx_pp - dx) / eps;
    c10::complex<T> d2yx = (dy_px - dy) / eps;
    c10::complex<T> d2yy = (dy_py - dy) / eps;
    c10::complex<T> d2yz = (dy_pz - dy) / eps;
    c10::complex<T> d2yp = (dy_pp - dy) / eps;
    c10::complex<T> d2zx = (dz_px - dz) / eps;
    c10::complex<T> d2zy = (dz_py - dz) / eps;
    c10::complex<T> d2zz = (dz_pz - dz) / eps;
    c10::complex<T> d2zp = (dz_pp - dz) / eps;
    c10::complex<T> d2px = (dp_px - dp) / eps;
    c10::complex<T> d2py = (dp_py - dp) / eps;
    c10::complex<T> d2pz = (dp_pz - dp) / eps;
    c10::complex<T> d2pp = (dp_pp - dp) / eps;

    c10::complex<T> grad_x = gradient * (gg_x * d2xx + gg_y * d2yx + gg_z * d2zx + gg_p * d2px);
    c10::complex<T> grad_y = gradient * (gg_x * d2xy + gg_y * d2yy + gg_z * d2zy + gg_p * d2py);
    c10::complex<T> grad_z = gradient * (gg_x * d2xz + gg_y * d2yz + gg_z * d2zz + gg_p * d2pz);
    c10::complex<T> grad_p = gradient * (gg_x * d2xp + gg_y * d2yp + gg_z * d2zp + gg_p * d2pp);

    return {grad_gradient, grad_x, grad_y, grad_z, grad_p};
}

} // namespace torchscience::kernel::special_functions
