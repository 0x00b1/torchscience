#pragma once

#include <c10/util/complex.h>
#include <cmath>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_j.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T incomplete_legendre_elliptic_integral_pi(T n, T phi, T m) {
    // Incomplete elliptic integral of the third kind Pi(n, phi, m)
    //
    // Mathematical definition:
    // Pi(n, phi, m) = integral from 0 to phi of
    //                 dtheta / [(1 - n*sin^2(theta)) * sqrt(1 - m*sin^2(theta))]
    //
    // Relation to Carlson integrals:
    // Pi(n, phi, m) = sin(phi) * R_F(cos^2(phi), 1-m*sin^2(phi), 1)
    //                 + (n/3) * sin^3(phi) * R_J(cos^2(phi), 1-m*sin^2(phi), 1, 1-n*sin^2(phi))
    //
    // where m is the parameter (m = k^2 with k being the modulus)
    // and n is the characteristic

    T sin_phi = std::sin(phi);
    T cos_phi = std::cos(phi);

    T sin2_phi = sin_phi * sin_phi;
    T cos2_phi = cos_phi * cos_phi;
    T sin3_phi = sin_phi * sin2_phi;

    T one = T(1);
    T one_minus_m_sin2 = one - m * sin2_phi;
    T one_minus_n_sin2 = one - n * sin2_phi;

    // R_F(cos^2(phi), 1-m*sin^2(phi), 1)
    T rf = carlson_elliptic_integral_r_f(cos2_phi, one_minus_m_sin2, one);

    // R_J(cos^2(phi), 1-m*sin^2(phi), 1, 1-n*sin^2(phi))
    T rj = carlson_elliptic_integral_r_j(cos2_phi, one_minus_m_sin2, one, one_minus_n_sin2);

    // Pi(n, phi, m) = sin(phi) * R_F + (n/3) * sin^3(phi) * R_J
    return sin_phi * rf + (n / T(3)) * sin3_phi * rj;
}

template <typename T>
c10::complex<T> incomplete_legendre_elliptic_integral_pi(
    c10::complex<T> n,
    c10::complex<T> phi,
    c10::complex<T> m
) {
    c10::complex<T> sin_phi = std::sin(phi);
    c10::complex<T> cos_phi = std::cos(phi);

    c10::complex<T> sin2_phi = sin_phi * sin_phi;
    c10::complex<T> cos2_phi = cos_phi * cos_phi;
    c10::complex<T> sin3_phi = sin_phi * sin2_phi;

    c10::complex<T> one(T(1), T(0));
    c10::complex<T> three(T(3), T(0));
    c10::complex<T> one_minus_m_sin2 = one - m * sin2_phi;
    c10::complex<T> one_minus_n_sin2 = one - n * sin2_phi;

    // R_F(cos^2(phi), 1-m*sin^2(phi), 1)
    c10::complex<T> rf = carlson_elliptic_integral_r_f(cos2_phi, one_minus_m_sin2, one);

    // R_J(cos^2(phi), 1-m*sin^2(phi), 1, 1-n*sin^2(phi))
    c10::complex<T> rj = carlson_elliptic_integral_r_j(cos2_phi, one_minus_m_sin2, one, one_minus_n_sin2);

    // Pi(n, phi, m) = sin(phi) * R_F + (n/3) * sin^3(phi) * R_J
    return sin_phi * rf + (n / three) * sin3_phi * rj;
}

} // namespace torchscience::kernel::special_functions
