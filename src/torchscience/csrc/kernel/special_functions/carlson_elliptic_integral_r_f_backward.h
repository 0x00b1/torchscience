#pragma once

#include <c10/util/complex.h>
#include <tuple>

#include "carlson_elliptic_integral_r_d.h"

namespace torchscience::kernel::special_functions {

template <typename T>
std::tuple<T, T, T> carlson_elliptic_integral_r_f_backward(
    T gradient,
    T x,
    T y,
    T z
) {
    // dR_F/dx = -R_D(y, z, x) / 6
    // dR_F/dy = -R_D(z, x, y) / 6
    // dR_F/dz = -R_D(x, y, z) / 6
    T dx = -carlson_elliptic_integral_r_d(y, z, x) / T(6);
    T dy = -carlson_elliptic_integral_r_d(z, x, y) / T(6);
    T dz = -carlson_elliptic_integral_r_d(x, y, z) / T(6);
    return {gradient * dx, gradient * dy, gradient * dz};
}

template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>>
carlson_elliptic_integral_r_f_backward(
    c10::complex<T> gradient,
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    c10::complex<T> dx = -carlson_elliptic_integral_r_d(y, z, x) / T(6);
    c10::complex<T> dy = -carlson_elliptic_integral_r_d(z, x, y) / T(6);
    c10::complex<T> dz = -carlson_elliptic_integral_r_d(x, y, z) / T(6);
    return {gradient * std::conj(dx), gradient * std::conj(dy), gradient * std::conj(dz)};
}

} // namespace torchscience::kernel::special_functions
