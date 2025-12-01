#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_g.h"

namespace torchscience::kernel::special_functions {

// Backward pass for the complete elliptic integral of the first kind K(m)
//
// The gradient of K(m) with respect to m is:
// dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
//
// where:
// - K(m) = R_F(0, 1-m, 1) is the complete elliptic integral of the first kind
// - E(m) = 2 * R_G(0, 1-m, 1) is the complete elliptic integral of the second kind
//
// Special cases:
// - At m = 0: dK/dm = pi/4 (can be derived from series expansion)
// - At m = 1: dK/dm -> infinity (singularity)

template <typename T>
T complete_legendre_elliptic_integral_k_backward(T gradient, T m) {
    const T eps = std::numeric_limits<T>::epsilon();

    // Handle the m = 0 case with the known limit
    // K(m) ~ pi/2 + (pi/8)m + O(m^2) as m -> 0
    // dK/dm ~ pi/4 + O(m) as m -> 0
    if (std::abs(m) < eps) {
        return gradient * static_cast<T>(M_PI) / T(4);
    }

    // Handle m = 1 case (singularity)
    if (std::abs(m - T(1)) < eps) {
        return gradient * std::numeric_limits<T>::infinity();
    }

    T one_minus_m = T(1) - m;

    // K(m) = R_F(0, 1-m, 1)
    T K = carlson_elliptic_integral_r_f(T(0), one_minus_m, T(1));

    // E(m) = 2 * R_G(0, 1-m, 1)
    T E = T(2) * carlson_elliptic_integral_r_g(T(0), one_minus_m, T(1));

    // dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
    T numerator = E - one_minus_m * K;
    T denominator = T(2) * m * one_minus_m;

    return gradient * numerator / denominator;
}

template <typename T>
c10::complex<T> complete_legendre_elliptic_integral_k_backward(
    c10::complex<T> gradient,
    c10::complex<T> m
) {
    const T eps = std::numeric_limits<T>::epsilon();
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> four(T(4), T(0));
    c10::complex<T> pi_val(static_cast<T>(M_PI), T(0));

    // Handle the m ~ 0 case
    if (std::abs(m) < eps) {
        c10::complex<T> deriv = pi_val / four;
        return gradient * std::conj(deriv);
    }

    // Handle m ~ 1 case (singularity)
    if (std::abs(m - one) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    c10::complex<T> one_minus_m = one - m;

    // K(m) = R_F(0, 1-m, 1)
    c10::complex<T> K = carlson_elliptic_integral_r_f(zero, one_minus_m, one);

    // E(m) = 2 * R_G(0, 1-m, 1)
    c10::complex<T> E = two * carlson_elliptic_integral_r_g(zero, one_minus_m, one);

    // dK/dm = [E(m) - (1-m)K(m)] / [2m(1-m)]
    c10::complex<T> numerator = E - one_minus_m * K;
    c10::complex<T> denominator = two * m * one_minus_m;

    c10::complex<T> deriv = numerator / denominator;

    // For complex inputs, PyTorch uses Wirtinger derivatives
    // The backward returns grad * conj(d/dz f(z)) for holomorphic f
    return gradient * std::conj(deriv);
}

} // namespace torchscience::kernel::special_functions
