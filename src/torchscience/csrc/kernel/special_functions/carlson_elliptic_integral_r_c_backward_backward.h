#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_f_backward_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_c_backward_backward(
    T gg_x,
    T gg_y,
    T gradient,
    T x,
    T y
) {
    // R_C(x, y) = R_F(x, y, y)
    // Chain rule through R_F
    auto [grad_gradient, grad_x, grad_y, grad_z] =
        carlson_elliptic_integral_r_f_backward_backward(gg_x, gg_y, gg_y, gradient, x, y, y);
    return {grad_gradient, grad_x, grad_y + grad_z};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_c_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y
) {
    auto [grad_gradient, grad_x, grad_y, grad_z] =
        carlson_elliptic_integral_r_f_backward_backward(gg_x, gg_y, gg_y, gradient, x, y, y);
    return {grad_gradient, grad_x, grad_y + grad_z};
}

} // namespace torchscience::kernel::special_functions
