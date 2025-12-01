#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arcsn(x, m) = inverse of sn(u, m)
//
// Mathematical definition:
// arcsn(x, m) = u such that sn(u, m) = x
//
// Expressed using Carlson's symmetric elliptic integral R_F:
// arcsn(x, m) = x * R_F(1 - x^2, 1 - m*x^2, 1)
//
// Domain:
// - x: real or complex, typically |x| <= 1 for real m with 0 <= m <= 1
// - m: elliptic parameter (conventionally 0 <= m <= 1)
//
// Special values:
// - arcsn(0, m) = 0 for all m
// - arcsn(1, m) = K(m) where K(m) is the complete elliptic integral of the first kind
// - arcsn(x, 0) = arcsin(x) (circular limit)
// - arcsn(x, 1) = arctanh(x) (hyperbolic limit)
//
// Algorithm:
// Uses the relationship arcsn(x, m) = x * R_F(1-x^2, 1-m*x^2, 1)
// where R_F is Carlson's symmetric elliptic integral of the first kind.

namespace detail {

template <typename T>
constexpr T inverse_jacobi_tolerance();

template <>
constexpr float inverse_jacobi_tolerance<float>() { return 1e-7f; }

template <>
constexpr double inverse_jacobi_tolerance<double>() { return 1e-15; }

template <>
inline c10::Half inverse_jacobi_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_sn(T x, T m) {
    const T tol = detail::inverse_jacobi_tolerance<T>();

    // Handle special case x = 0: arcsn(0, m) = 0
    if (std::abs(x) < tol) {
        return T(0);
    }

    // Handle special case m = 0: arcsn(x, 0) = arcsin(x)
    if (std::abs(m) < tol) {
        return std::asin(x);
    }

    // Handle special case m = 1: arcsn(x, 1) = arctanh(x)
    if (std::abs(m - T(1)) < tol) {
        return std::atanh(x);
    }

    // General case: arcsn(x, m) = x * R_F(1 - x^2, 1 - m*x^2, 1)
    T x2 = x * x;
    T arg1 = T(1) - x2;
    T arg2 = T(1) - m * x2;
    T arg3 = T(1);

    // Handle edge cases where arguments might be negative
    // For real x with |x| > 1 or complex x, we may get negative arguments
    // The R_F function handles complex arguments via its complex overload

    return x * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

template <typename T>
c10::complex<T> inverse_jacobi_elliptic_sn(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_tolerance<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));

    // Handle special case x = 0: arcsn(0, m) = 0
    if (std::abs(x) < tol) {
        return zero;
    }

    // Handle special case m = 0: arcsn(x, 0) = arcsin(x)
    if (std::abs(m) < tol) {
        return std::asin(x);
    }

    // Handle special case m = 1: arcsn(x, 1) = arctanh(x)
    if (std::abs(m - one) < tol) {
        return std::atanh(x);
    }

    // General case: arcsn(x, m) = x * R_F(1 - x^2, 1 - m*x^2, 1)
    c10::complex<T> x2 = x * x;
    c10::complex<T> arg1 = one - x2;
    c10::complex<T> arg2 = one - m * x2;
    c10::complex<T> arg3 = one;

    return x * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

} // namespace torchscience::kernel::special_functions
