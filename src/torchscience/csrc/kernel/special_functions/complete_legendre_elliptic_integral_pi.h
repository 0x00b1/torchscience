#pragma once

#include <c10/util/complex.h>

#include "carlson_elliptic_integral_r_f.h"
#include "carlson_elliptic_integral_r_j.h"

namespace torchscience::kernel::special_functions {

template <typename T>
T complete_legendre_elliptic_integral_pi(T n, T m) {
    // Complete elliptic integral of the third kind Pi(n, m)
    //
    // Mathematical definition:
    // Pi(n, m) = integral from 0 to pi/2 of
    //            dtheta / [(1 - n*sin^2(theta)) * sqrt(1 - m*sin^2(theta))]
    //
    // Relation to Carlson integrals:
    // Pi(n, m) = R_F(0, 1-m, 1) + (n/3) * R_J(0, 1-m, 1, 1-n)
    //
    // where m is the parameter (m = k^2 with k being the modulus)
    // and n is the characteristic

    T one = T(1);
    T zero = T(0);
    T one_minus_m = one - m;
    T one_minus_n = one - n;

    // R_F(0, 1-m, 1)
    T rf = carlson_elliptic_integral_r_f(zero, one_minus_m, one);

    // R_J(0, 1-m, 1, 1-n)
    T rj = carlson_elliptic_integral_r_j(zero, one_minus_m, one, one_minus_n);

    // Pi(n, m) = R_F + (n/3) * R_J
    return rf + (n / T(3)) * rj;
}

template <typename T>
c10::complex<T> complete_legendre_elliptic_integral_pi(
    c10::complex<T> n,
    c10::complex<T> m
) {
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));
    c10::complex<T> three(T(3), T(0));
    c10::complex<T> one_minus_m = one - m;
    c10::complex<T> one_minus_n = one - n;

    // R_F(0, 1-m, 1)
    c10::complex<T> rf = carlson_elliptic_integral_r_f(zero, one_minus_m, one);

    // R_J(0, 1-m, 1, 1-n)
    c10::complex<T> rj = carlson_elliptic_integral_r_j(zero, one_minus_m, one, one_minus_n);

    // Pi(n, m) = R_F + (n/3) * R_J
    return rf + (n / three) * rj;
}

} // namespace torchscience::kernel::special_functions
