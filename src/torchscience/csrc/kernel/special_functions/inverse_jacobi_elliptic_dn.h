#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arcdn(x, m) = inverse of dn(u, m)
//
// Mathematical definition:
// arcdn(x, m) = u such that dn(u, m) = x
//
// Expressed using Carlson's symmetric elliptic integral R_F:
// arcdn(x, m) = sqrt((1-x^2)/m) * R_F((m + x^2 - 1)/m, x^2, 1)
//
// Derivation:
// dn(u, m) = sqrt(1 - m*sn^2(u, m))
// If dn(u, m) = x, then sn^2(u, m) = (1 - x^2)/m
// Let sn(u, m) = sin(phi), so sin^2(phi) = (1 - x^2)/m
// cos^2(phi) = 1 - sin^2(phi) = (m + x^2 - 1)/m
// 1 - m*sin^2(phi) = x^2
// u = F(phi, m) = sin(phi) * R_F(cos^2(phi), 1 - m*sin^2(phi), 1)
//   = sqrt((1-x^2)/m) * R_F((m + x^2 - 1)/m, x^2, 1)
//
// Domain:
// - x: real or complex, typically sqrt(1-m) <= x <= 1 for real m with 0 <= m <= 1
// - m: elliptic parameter (conventionally 0 < m <= 1, m != 0)
//
// Special values:
// - arcdn(1, m) = 0 for all m (since dn(0, m) = 1)
// - arcdn(sqrt(1-m), m) = K(m) where K(m) is the complete elliptic integral
// - arcdn(x, 0) = undefined (dn(u,0) = 1 for all u, so no unique inverse)
// - arcdn(x, 1) = arcsech(x) (hyperbolic limit, same as arccn for m=1)
//
// Algorithm:
// Uses the relationship arcdn(x, m) = sqrt((1-x^2)/m) * R_F((m+x^2-1)/m, x^2, 1)
// where R_F is Carlson's symmetric elliptic integral of the first kind.

namespace detail {

template <typename T>
constexpr T inverse_jacobi_dn_tolerance();

template <>
constexpr float inverse_jacobi_dn_tolerance<float>() { return 1e-7f; }

template <>
constexpr double inverse_jacobi_dn_tolerance<double>() { return 1e-15; }

template <>
inline c10::Half inverse_jacobi_dn_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_dn_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_dn(T x, T m) {
    const T tol = detail::inverse_jacobi_dn_tolerance<T>();

    // Handle special case x = 1: arcdn(1, m) = 0
    if (std::abs(x - T(1)) < tol) {
        return T(0);
    }

    // Handle special case m = 0: arcdn(x, 0) is undefined
    // dn(u, 0) = 1 for all u, so there's no unique inverse
    // Return NaN to indicate undefined
    if (std::abs(m) < tol) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // Handle special case m = 1: arcdn(x, 1) = arcsech(x)
    // For m = 1, dn(u, 1) = sech(u), so arcdn(x, 1) = arcsech(x)
    if (std::abs(m - T(1)) < tol) {
        // arcsech(x) = log((1 + sqrt(1 - x^2)) / x) for 0 < x <= 1
        T sqrt_term = std::sqrt(T(1) - x * x);
        return std::log((T(1) + sqrt_term) / x);
    }

    // General case: arcdn(x, m) = sqrt((1-x^2)/m) * R_F((m+x^2-1)/m, x^2, 1)
    T x2 = x * x;
    T one_minus_x2 = T(1) - x2;

    // sin(phi) = sqrt((1 - x^2) / m)
    T sin_phi = std::sqrt(one_minus_x2 / m);

    // cos^2(phi) = (m + x^2 - 1) / m
    T cos_phi_sq = (m + x2 - T(1)) / m;

    // 1 - m*sin^2(phi) = x^2
    T arg1 = cos_phi_sq;
    T arg2 = x2;
    T arg3 = T(1);

    return sin_phi * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

template <typename T>
c10::complex<T> inverse_jacobi_elliptic_dn(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_dn_tolerance<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));

    // Handle special case x = 1: arcdn(1, m) = 0
    if (std::abs(x - one) < tol) {
        return zero;
    }

    // Handle special case m = 0: arcdn(x, 0) is undefined
    if (std::abs(m) < tol) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
    }

    // Handle special case m = 1: arcdn(x, 1) = arcsech(x)
    if (std::abs(m - one) < tol) {
        c10::complex<T> sqrt_term = std::sqrt(one - x * x);
        return std::log((one + sqrt_term) / x);
    }

    // General case: arcdn(x, m) = sqrt((1-x^2)/m) * R_F((m+x^2-1)/m, x^2, 1)
    c10::complex<T> x2 = x * x;
    c10::complex<T> one_minus_x2 = one - x2;

    // sin(phi) = sqrt((1 - x^2) / m)
    c10::complex<T> sin_phi = std::sqrt(one_minus_x2 / m);

    // cos^2(phi) = (m + x^2 - 1) / m
    c10::complex<T> cos_phi_sq = (m + x2 - one) / m;

    // 1 - m*sin^2(phi) = x^2
    c10::complex<T> arg1 = cos_phi_sq;
    c10::complex<T> arg2 = x2;
    c10::complex<T> arg3 = one;

    return sin_phi * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

} // namespace torchscience::kernel::special_functions
