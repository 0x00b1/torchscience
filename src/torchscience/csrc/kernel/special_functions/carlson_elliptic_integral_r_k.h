#pragma once

#include <c10/util/complex.h>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T carlson_elliptic_integral_r_k(T x, T y) {
    // R_K(x, y) = R_F(0, x, y)
    // Complete elliptic integral of the first kind in Carlson form
    return carlson_elliptic_integral_r_f(T(0), x, y);
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_k(c10::complex<T> x, c10::complex<T> y) {
    return carlson_elliptic_integral_r_f(c10::complex<T>(T(0), T(0)), x, y);
}

} // namespace torchscience::kernel::special_functions
