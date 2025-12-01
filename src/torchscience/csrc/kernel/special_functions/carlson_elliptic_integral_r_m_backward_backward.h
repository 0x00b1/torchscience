#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_m.h"
#include "carlson_elliptic_integral_r_m_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T, T> carlson_elliptic_integral_r_m_backward_backward(
    T gg_x,
    T gg_y,
    T gg_z,
    T gradient,
    T x,
    T y,
    T z
) {
    // Numerical approximation for second derivatives
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());

    auto [dx, dy, dz] = carlson_elliptic_integral_r_m_backward(T(1), x, y, z);

    // Gradient w.r.t. incoming gradient
    T grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz;

    // Second derivatives via finite differences
    auto [dx_px, dy_px, dz_px] = carlson_elliptic_integral_r_m_backward(T(1), x + eps, y, z);
    auto [dx_py, dy_py, dz_py] = carlson_elliptic_integral_r_m_backward(T(1), x, y + eps, z);
    auto [dx_pz, dy_pz, dz_pz] = carlson_elliptic_integral_r_m_backward(T(1), x, y, z + eps);

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
carlson_elliptic_integral_r_m_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gg_z,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    T eps = std::sqrt(std::numeric_limits<T>::epsilon());
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> eps_c(eps, T(0));

    auto [dx, dy, dz] = carlson_elliptic_integral_r_m_backward(one, x, y, z);

    c10::complex<T> grad_gradient = gg_x * dx + gg_y * dy + gg_z * dz;

    auto [dx_px, dy_px, dz_px] = carlson_elliptic_integral_r_m_backward(one, x + eps_c, y, z);
    auto [dx_py, dy_py, dz_py] = carlson_elliptic_integral_r_m_backward(one, x, y + eps_c, z);
    auto [dx_pz, dy_pz, dz_pz] = carlson_elliptic_integral_r_m_backward(one, x, y, z + eps_c);

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
