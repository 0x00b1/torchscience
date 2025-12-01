#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_d.h"
#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_d_backward(
    T gradient,
    T x,
    T y,
    T z
) {
    // dR_D/dx = (R_D(y,z,x) - R_D(x,y,z)) / (2(x-z))
    // dR_D/dy = (R_D(z,x,y) - R_D(x,y,z)) / (2(y-z))
    // dR_D/dz = -(3R_F(x,y,z) - x*R_D(y,z,x) - y*R_D(z,x,y)) / (2z)

    T rd_xyz = carlson_elliptic_integral_r_d(x, y, z);
    T rd_yzx = carlson_elliptic_integral_r_d(y, z, x);
    T rd_zxy = carlson_elliptic_integral_r_d(z, x, y);
    T rf_xyz = carlson_elliptic_integral_r_f(x, y, z);

    T dx = (rd_yzx - rd_xyz) / (T(2) * (x - z));
    T dy = (rd_zxy - rd_xyz) / (T(2) * (y - z));
    T dz = -(T(3) * rf_xyz - x * rd_yzx - y * rd_zxy) / (T(2) * z);

    return {gradient * dx, gradient * dy, gradient * dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_d_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    c10::complex<T> rd_xyz = carlson_elliptic_integral_r_d(x, y, z);
    c10::complex<T> rd_yzx = carlson_elliptic_integral_r_d(y, z, x);
    c10::complex<T> rd_zxy = carlson_elliptic_integral_r_d(z, x, y);
    c10::complex<T> rf_xyz = carlson_elliptic_integral_r_f(x, y, z);

    c10::complex<T> dx = (rd_yzx - rd_xyz) / (c10::complex<T>(T(2), T(0)) * (x - z));
    c10::complex<T> dy = (rd_zxy - rd_xyz) / (c10::complex<T>(T(2), T(0)) * (y - z));
    c10::complex<T> dz = -(c10::complex<T>(T(3), T(0)) * rf_xyz - x * rd_yzx - y * rd_zxy)
                         / (c10::complex<T>(T(2), T(0)) * z);

    return {gradient * std::conj(dx), gradient * std::conj(dy), gradient * std::conj(dz)};
}

} // namespace torchscience::kernel::special_functions
