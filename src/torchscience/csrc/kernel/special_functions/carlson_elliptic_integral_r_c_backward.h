#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_f_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> carlson_elliptic_integral_r_c_backward(
    T gradient,
    T x,
    T y
) {
    // R_C(x, y) = R_F(x, y, y)
    // dR_C/dx = dR_F/dx
    // dR_C/dy = dR_F/dy + dR_F/dz
    auto [dx, dy, dz] = carlson_elliptic_integral_r_f_backward(gradient, x, y, y);
    return {dx, dy + dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_c_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y
) {
    auto [dx, dy, dz] = carlson_elliptic_integral_r_f_backward(gradient, x, y, y);
    return {dx, dy + dz};
}

} // namespace torchscience::kernel::special_functions
