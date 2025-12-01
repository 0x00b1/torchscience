#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_d.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T carlson_elliptic_integral_r_e(T x, T y, T z) {
    // Carlson's elliptic integral R_E(x, y, z)
    //
    // Mathematical definition:
    // R_E(x,y,z) = (3/2) * integral from 0 to infinity of
    //              t dt / [(t+z) * sqrt((t+x)(t+y)(t+z))]
    //
    // Key relationship (used for implementation):
    // R_E(x,y,z) = (3/2)*z*R_D(x,y,z) + sqrt(xy/z)
    //
    // Alternative relationship:
    // R_E(x,y,z) = R_D(y,z,x) + R_D(z,x,y) + 3*sqrt(xyz/(xy+xz+yz))
    //
    // R_E is NOT symmetric in all three arguments - z plays a distinguished role.

    // Handle the case z = 0 specially
    if (z == T(0)) {
        // When z = 0, R_E(x,y,0) diverges unless handled specially
        // For numerical stability, return a large value or NaN
        return std::numeric_limits<T>::infinity();
    }

    // Use the formula: R_E(x,y,z) = (3/2)*z*R_D(x,y,z) + sqrt(xy/z)
    T rd = carlson_elliptic_integral_r_d(x, y, z);
    T term1 = T(1.5) * z * rd;
    T term2 = std::sqrt(x * y / z);

    return term1 + term2;
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_e(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    // For complex arguments, we use the same formula:
    // R_E(x,y,z) = (3/2)*z*R_D(x,y,z) + sqrt(xy/z)

    c10::complex<T> three_half(T(1.5), T(0));

    // Handle z = 0 case
    T abs_z = std::abs(z);
    if (abs_z < std::numeric_limits<T>::epsilon()) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> rd = carlson_elliptic_integral_r_d(x, y, z);
    c10::complex<T> term1 = three_half * z * rd;
    c10::complex<T> term2 = std::sqrt(x * y / z);

    return term1 + term2;
}

} // namespace torchscience::kernel::special_functions
