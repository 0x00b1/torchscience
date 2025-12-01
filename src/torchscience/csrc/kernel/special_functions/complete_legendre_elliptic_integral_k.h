#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

// Complete elliptic integral of the first kind K(m)
//
// Mathematical definition:
// K(m) = integral from 0 to pi/2 of d(theta) / sqrt(1 - m * sin^2(theta))
//
// This is the Legendre form with parameter m (the "parameter convention").
// Note: Some references use the modulus k where m = k^2.
//
// Key relationship to Carlson form:
// K(m) = R_F(0, 1-m, 1)
//
// Domain:
// - For real m: m < 1 (singularity at m = 1 where K -> infinity)
// - For complex m: entire complex plane with branch cut at [1, infinity)
//
// Special values:
// - K(0) = pi/2
// - K(1) = infinity (logarithmic singularity)

template <typename T>
T complete_legendre_elliptic_integral_k(T m) {
    // K(m) = R_F(0, 1-m, 1)
    return carlson_elliptic_integral_r_f(T(0), T(1) - m, T(1));
}

template <typename T>
c10::complex<T> complete_legendre_elliptic_integral_k(c10::complex<T> m) {
    // K(m) = R_F(0, 1-m, 1)
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));
    return carlson_elliptic_integral_r_f(zero, one - m, one);
}

} // namespace torchscience::kernel::special_functions
