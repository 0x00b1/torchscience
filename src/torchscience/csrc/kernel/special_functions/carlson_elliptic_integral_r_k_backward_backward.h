#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_f_backward_backward.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_k_backward_backward(
    T gg_x,
    T gg_y,
    T gradient,
    T x,
    T y
) {
    // R_K(x, y) = R_F(0, x, y)
    // Chain rule through R_F
    // gg_x corresponds to R_F's y, gg_y corresponds to R_F's z
    auto [grad_gradient, grad_0, grad_x, grad_y] =
        carlson_elliptic_integral_r_f_backward_backward(T(0), gg_x, gg_y, gradient, T(0), x, y);
    return {grad_gradient, grad_x, grad_y};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_k_backward_backward(
    c10::complex<T> gg_x,
    c10::complex<T> gg_y,
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y
) {
    auto [grad_gradient, grad_0, grad_x, grad_y] =
        carlson_elliptic_integral_r_f_backward_backward(
            c10::complex<T>(T(0), T(0)), gg_x, gg_y, gradient,
            c10::complex<T>(T(0), T(0)), x, y);
    return {grad_gradient, grad_x, grad_y};
}

} // namespace torchscience::kernel::special_functions
