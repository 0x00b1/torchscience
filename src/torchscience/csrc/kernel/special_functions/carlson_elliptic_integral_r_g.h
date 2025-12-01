#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_d.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T carlson_elliptic_integral_r_g(T x, T y, T z) {
    // Carlson's elliptic integral R_G(x, y, z)
    //
    // Mathematical definition:
    // R_G(x,y,z) = (1/4pi) * integral over sphere of
    //              sqrt(x*cos^2(theta)*sin^2(phi) + y*sin^2(theta)*sin^2(phi) + z*cos^2(phi))
    //              * sin(phi) d(theta) d(phi)
    //
    // Key relationship (used for implementation):
    // R_G(x,y,z) = (1/2)[z*R_F(x,y,z) - (1/3)(x-z)(y-z)*R_D(x,y,z) + sqrt(xy/z)]
    //
    // R_G is fully symmetric in x, y, z.

    // Handle the case z = 0 using symmetry: R_G(x, y, 0) = R_G(x, 0, y) = R_G(0, x, y)
    // We use the formula with the largest argument in the z position to avoid division by zero
    // and to improve numerical stability.

    // Find the maximum value and reorder arguments
    T a = x, b = y, c = z;

    // Sort so that c >= b >= a (c is largest)
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);

    // Now c is the largest. If c = 0, all are zero.
    if (c == T(0)) {
        return T(0);
    }

    // Use the formula: R_G(a,b,c) = (1/2)[c*R_F(a,b,c) - (1/3)(a-c)(b-c)*R_D(a,b,c) + sqrt(ab/c)]
    T rf = carlson_elliptic_integral_r_f(a, b, c);
    T rd = carlson_elliptic_integral_r_d(a, b, c);

    T term1 = c * rf;
    T term2 = (a - c) * (b - c) * rd / T(3);
    T term3 = std::sqrt(a * b / c);

    return (term1 - term2 + term3) / T(2);
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_g(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    // For complex arguments, we use the same formula but need to handle
    // the branch cut selection more carefully.
    //
    // R_G(x,y,z) = (1/2)[z*R_F(x,y,z) - (1/3)(x-z)(y-z)*R_D(x,y,z) + sqrt(xy/z)]

    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> three(T(3), T(0));

    // Handle z = 0 case by using symmetry
    // We check if |z| is very small and swap with a non-zero argument
    T abs_x = std::abs(x);
    T abs_y = std::abs(y);
    T abs_z = std::abs(z);

    c10::complex<T> a = x, b = y, c = z;
    T abs_c = abs_z;

    // Put the largest magnitude in c position
    if (abs_x > abs_z || abs_y > abs_z) {
        if (abs_x >= abs_y) {
            // x has largest magnitude
            c = x;
            a = y;
            b = z;
            abs_c = abs_x;
        } else {
            // y has largest magnitude
            c = y;
            a = x;
            b = z;
            abs_c = abs_y;
        }
    }

    // If all are zero (or essentially zero)
    if (abs_c < std::numeric_limits<T>::epsilon()) {
        return c10::complex<T>(T(0), T(0));
    }

    c10::complex<T> rf = carlson_elliptic_integral_r_f(a, b, c);
    c10::complex<T> rd = carlson_elliptic_integral_r_d(a, b, c);

    c10::complex<T> term1 = c * rf;
    c10::complex<T> term2 = (a - c) * (b - c) * rd / three;
    c10::complex<T> term3 = std::sqrt(a * b / c);

    return (term1 - term2 + term3) / two;
}

} // namespace torchscience::kernel::special_functions
