#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_d.h"

namespace torchscience::kernel::special_functions {

// Incomplete elliptic integral of the second kind E(phi, m)
//
// Mathematical definition:
// E(phi, m) = integral from 0 to phi of sqrt(1 - m * sin^2(theta)) d(theta)
//
// Key relationship using Carlson symmetric integrals:
// E(phi, m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)
//           - (m/3) * sin^3(phi) * R_D(cos^2(phi), 1 - m*sin^2(phi), 1)
//
// Domain:
// - For real inputs: phi in R, m <= 1 for real results
// - For complex inputs: entire complex plane
//
// Special values:
// - E(0, m) = 0
// - E(pi/2, m) = complete E(m)

template <typename T>
T incomplete_legendre_elliptic_integral_e(T phi, T m) {
    // Handle edge case phi = 0
    if (phi == T(0)) {
        return T(0);
    }

    T sin_phi = std::sin(phi);
    T cos_phi = std::cos(phi);

    T sin2_phi = sin_phi * sin_phi;
    T cos2_phi = cos_phi * cos_phi;
    T sin3_phi = sin_phi * sin2_phi;

    // Arguments for Carlson integrals
    T x = cos2_phi;
    T y = T(1) - m * sin2_phi;
    T z = T(1);

    // E(phi, m) = sin(phi) * R_F(x, y, z) - (m/3) * sin^3(phi) * R_D(x, y, z)
    T rf = carlson_elliptic_integral_r_f(x, y, z);
    T rd = carlson_elliptic_integral_r_d(x, y, z);

    return sin_phi * rf - (m / T(3)) * sin3_phi * rd;
}

template <typename T>
c10::complex<T> incomplete_legendre_elliptic_integral_e(
    c10::complex<T> phi,
    c10::complex<T> m
) {
    // Handle edge case phi = 0
    if (std::abs(phi) < T(1e-14)) {
        return c10::complex<T>(T(0), T(0));
    }

    c10::complex<T> sin_phi = std::sin(phi);
    c10::complex<T> cos_phi = std::cos(phi);

    c10::complex<T> sin2_phi = sin_phi * sin_phi;
    c10::complex<T> cos2_phi = cos_phi * cos_phi;
    c10::complex<T> sin3_phi = sin_phi * sin2_phi;

    // Arguments for Carlson integrals
    c10::complex<T> x = cos2_phi;
    c10::complex<T> y = c10::complex<T>(T(1), T(0)) - m * sin2_phi;
    c10::complex<T> z(T(1), T(0));

    // E(phi, m) = sin(phi) * R_F(x, y, z) - (m/3) * sin^3(phi) * R_D(x, y, z)
    c10::complex<T> rf = carlson_elliptic_integral_r_f(x, y, z);
    c10::complex<T> rd = carlson_elliptic_integral_r_d(x, y, z);

    c10::complex<T> three(T(3), T(0));
    return sin_phi * rf - (m / three) * sin3_phi * rd;
}

} // namespace torchscience::kernel::special_functions
