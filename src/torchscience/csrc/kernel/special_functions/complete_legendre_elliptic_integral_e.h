#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_g.h"

namespace torchscience::kernel::special_functions {

// Complete elliptic integral of the second kind E(m)
//
// Mathematical definition:
// E(m) = integral from 0 to pi/2 of sqrt(1 - m * sin^2(theta)) d(theta)
//
// Key relationship:
// E(m) = 2 * R_G(0, 1-m, 1)
//
// where R_G is Carlson's symmetric elliptic integral.
//
// Domain:
// - For real m: m <= 1 (singularity at m = 1)
// - For complex m: entire complex plane
//
// Special values:
// - E(0) = pi/2
// - E(1) = 1

template <typename T>
T complete_legendre_elliptic_integral_e(T m) {
    // E(m) = 2 * R_G(0, 1-m, 1)
    T zero = T(0);
    T one = T(1);
    T one_minus_m = one - m;

    return T(2) * carlson_elliptic_integral_r_g(zero, one_minus_m, one);
}

template <typename T>
c10::complex<T> complete_legendre_elliptic_integral_e(c10::complex<T> m) {
    // E(m) = 2 * R_G(0, 1-m, 1)
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> one_minus_m = one - m;

    return two * carlson_elliptic_integral_r_g(zero, one_minus_m, one);
}

} // namespace torchscience::kernel::special_functions
