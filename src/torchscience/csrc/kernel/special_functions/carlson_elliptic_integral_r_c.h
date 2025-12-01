#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T carlson_elliptic_integral_r_c(T x, T y) {
    return carlson_elliptic_integral_r_f(x, y, y);
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_c(c10::complex<T> x, c10::complex<T> y) {
    return carlson_elliptic_integral_r_f(x, y, y);
}

} // namespace torchscience::kernel::special_functions
