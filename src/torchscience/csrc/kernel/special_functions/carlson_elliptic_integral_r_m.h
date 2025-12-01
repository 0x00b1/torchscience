#pragma once

#include <c10/util/complex.h>
#include <c10/util/MathConstants.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_j.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T carlson_elliptic_integral_r_m(T x, T y, T z) {
    // Carlson's elliptic integral R_M(x, y, z)
    //
    // Mathematical definition:
    // R_M(x,y,z) = (1/6)[(x+y-2z)R_F(x,y,z) + z*R_J(x,y,z,z) + 3*sqrt(xyz/(x+y-z))]
    //
    // Special case when z = 0:
    // R_M(x,y,0) = (pi/4)*(x+y)/sqrt(xy)
    //
    // R_M is related to the elliptic integral of the third kind.

    const T pi = T(c10::pi<double>);
    // Use a small but practical threshold for detecting near-zero
    const T small_eps = T(1e-15);
    // Use a larger threshold for detecting when the third term is degenerate
    const T denom_eps = T(1e-10);

    // Handle z = 0 special case
    if (std::abs(z) < small_eps) {
        T xy = x * y;
        if (std::abs(xy) < small_eps) {
            // Both x*y ~ 0, return 0 or handle degenerate case
            return T(0);
        }
        return (pi / T(4)) * (x + y) / std::sqrt(xy);
    }

    // General case: use the relationship to R_F and R_J
    // R_M(x,y,z) = (1/6)[(x+y-2z)R_F(x,y,z) + z*R_J(x,y,z,z) + 3*sqrt(xyz/(x+y-z))]

    T rf = carlson_elliptic_integral_r_f(x, y, z);
    T rj = carlson_elliptic_integral_r_j(x, y, z, z);

    T sum_xy_minus_2z = x + y - T(2) * z;
    T term1 = sum_xy_minus_2z * rf;
    T term2 = z * rj;

    // For term3: sqrt(xyz / (x+y-z))
    // The denominator (x + y - z) can be near zero causing numerical issues.
    // Also need to handle negative values inside sqrt for real numbers.
    T denom = x + y - z;
    T term3;
    T numer = x * y * z;
    if (std::abs(denom) < denom_eps || numer / denom < T(0)) {
        // When x + y - z ~ 0 or the ratio is negative, the term is singular/undefined
        // Return 0 for this term (this is a limitation of the real-valued formula)
        term3 = T(0);
    } else {
        term3 = T(3) * std::sqrt(numer / denom);
    }

    return (term1 + term2 + term3) / T(6);
}

template <typename T>
c10::complex<T> carlson_elliptic_integral_r_m(
    c10::complex<T> x,
    c10::complex<T> y,
    c10::complex<T> z
) {
    // For complex arguments, we use the same formula with appropriate
    // complex arithmetic.
    //
    // R_M(x,y,z) = (1/6)[(x+y-2z)R_F(x,y,z) + z*R_J(x,y,z,z) + 3*sqrt(xyz/(x+y-z))]

    const T pi = c10::pi<T>;
    // Use a small but practical threshold for detecting near-zero
    const T small_eps = T(1e-15);
    // Use a larger threshold for detecting when the third term is degenerate
    const T denom_eps = T(1e-10);

    c10::complex<T> two(T(2), T(0));
    c10::complex<T> three(T(3), T(0));
    c10::complex<T> four(T(4), T(0));
    c10::complex<T> six(T(6), T(0));
    c10::complex<T> pi_c(pi, T(0));

    // Handle z = 0 special case
    if (std::abs(z) < small_eps) {
        c10::complex<T> xy = x * y;
        if (std::abs(xy) < small_eps) {
            return c10::complex<T>(T(0), T(0));
        }
        return (pi_c / four) * (x + y) / std::sqrt(xy);
    }

    // General case
    c10::complex<T> rf = carlson_elliptic_integral_r_f(x, y, z);
    c10::complex<T> rj = carlson_elliptic_integral_r_j(x, y, z, z);

    c10::complex<T> sum_xy_minus_2z = x + y - two * z;
    c10::complex<T> term1 = sum_xy_minus_2z * rf;
    c10::complex<T> term2 = z * rj;

    // For term3: sqrt(xyz / (x+y-z))
    // For complex numbers, sqrt handles negative real parts gracefully
    c10::complex<T> denom = x + y - z;
    c10::complex<T> term3;
    if (std::abs(denom) < denom_eps) {
        term3 = c10::complex<T>(T(0), T(0));
    } else {
        term3 = three * std::sqrt(x * y * z / denom);
    }

    return (term1 + term2 + term3) / six;
}

} // namespace torchscience::kernel::special_functions
