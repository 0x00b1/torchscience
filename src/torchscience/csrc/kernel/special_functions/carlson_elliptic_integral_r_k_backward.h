#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_f_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T> carlson_elliptic_integral_r_k_backward(
    T gradient,
    T x,
    T y
) {
    // R_K(x, y) = R_F(0, x, y)
    // dR_K/dx = dR_F/dy (at R_F(0, x, y))
    // dR_K/dy = dR_F/dz (at R_F(0, x, y))
    auto [d0, dx, dy] = carlson_elliptic_integral_r_f_backward(gradient, T(0), x, y);
    return {dx, dy};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_k_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y
) {
    auto [d0, dx, dy] = carlson_elliptic_integral_r_f_backward(
        gradient, c10::complex<T>(T(0), T(0)), x, y);
    return {dx, dy};
}

} // namespace torchscience::kernel::special_functions
