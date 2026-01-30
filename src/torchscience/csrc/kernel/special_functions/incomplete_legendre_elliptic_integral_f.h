#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

// Incomplete elliptic integral of the first kind F(phi, m)
//
// Mathematical definition:
// F(phi, m) = integral from 0 to phi of 1 / sqrt(1 - m * sin^2(theta)) d(theta)
//
// Key relationship using Carlson symmetric integrals:
// F(phi, m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)
//
// Domain:
// - For real inputs: phi in R, 0 <= m <= 1 for real results
// - For complex inputs: entire complex plane
//
// Special values:
// - F(0, m) = 0
// - F(pi/2, m) = complete K(m)
// - F(-phi, m) = -F(phi, m) (odd in phi)
// - F(phi, 0) = phi
// - F(phi, 1) = arctanh(sin(phi)) for 0 <= phi < pi/2

template <typename T>
T incomplete_legendre_elliptic_integral_f(T phi, T m) {
    // Handle edge case phi = 0
    if (phi == T(0)) {
        return T(0);
    }

    // Handle special case m = 0: F(phi, 0) = phi
    if (m == T(0)) {
        return phi;
    }

    // Handle special case m = 1: F(phi, 1) = arctanh(sin(phi))
    // This is only valid for |phi| < pi/2
    if (m == T(1)) {
        T sin_phi = std::sin(phi);
        return std::atanh(sin_phi);
    }

    T sin_phi = std::sin(phi);
    T cos_phi = std::cos(phi);

    T sin2_phi = sin_phi * sin_phi;
    T cos2_phi = cos_phi * cos_phi;

    // Arguments for Carlson integral R_F
    T x = cos2_phi;
    T y = T(1) - m * sin2_phi;
    T z = T(1);

    // F(phi, m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)
    T rf = carlson_elliptic_integral_r_f(x, y, z);

    return sin_phi * rf;
}

template <typename T>
c10::complex<T> incomplete_legendre_elliptic_integral_f(
    c10::complex<T> phi,
    c10::complex<T> m
) {
    // Handle edge case phi = 0
    if (std::abs(phi) < T(1e-14)) {
        return c10::complex<T>(T(0), T(0));
    }

    // Handle special case m = 0: F(phi, 0) = phi
    if (std::abs(m) < T(1e-14)) {
        return phi;
    }

    c10::complex<T> sin_phi = std::sin(phi);
    c10::complex<T> cos_phi = std::cos(phi);

    c10::complex<T> sin2_phi = sin_phi * sin_phi;
    c10::complex<T> cos2_phi = cos_phi * cos_phi;

    // Arguments for Carlson integral R_F
    c10::complex<T> x = cos2_phi;
    c10::complex<T> y = c10::complex<T>(T(1), T(0)) - m * sin2_phi;
    c10::complex<T> z(T(1), T(0));

    // F(phi, m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)
    c10::complex<T> rf = carlson_elliptic_integral_r_f(x, y, z);

    return sin_phi * rf;
}

} // namespace torchscience::kernel::special_functions
