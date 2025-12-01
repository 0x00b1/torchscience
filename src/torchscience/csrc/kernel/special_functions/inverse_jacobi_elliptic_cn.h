#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

#include "carlson_elliptic_integral_r_f.h"

namespace torchscience::kernel::special_functions {

// Inverse Jacobi elliptic function arccn(x, m) = inverse of cn(u, m)
//
// Mathematical definition:
// arccn(x, m) = u such that cn(u, m) = x
//
// Expressed using Carlson's symmetric elliptic integral R_F:
// arccn(x, m) = sqrt(1 - x^2) * R_F(x^2, 1 - m + m*x^2, 1)
//
// Domain:
// - x: real or complex, typically |x| <= 1 for real m with 0 <= m <= 1
// - m: elliptic parameter (conventionally 0 <= m <= 1)
//
// Special values:
// - arccn(1, m) = 0 for all m (since cn(0, m) = 1)
// - arccn(0, m) = K(m) where K(m) is the complete elliptic integral of the first kind
// - arccn(x, 0) = arccos(x) (circular limit)
// - arccn(x, 1) = arcsech(x) (hyperbolic limit)
//
// Algorithm:
// Uses the relationship arccn(x, m) = sqrt(1-x^2) * R_F(x^2, 1-m+m*x^2, 1)
// where R_F is Carlson's symmetric elliptic integral of the first kind.

namespace detail {

template <typename T>
constexpr T inverse_jacobi_cn_tolerance();

template <>
constexpr float inverse_jacobi_cn_tolerance<float>() { return 1e-7f; }

template <>
constexpr double inverse_jacobi_cn_tolerance<double>() { return 1e-15; }

template <>
inline c10::Half inverse_jacobi_cn_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 inverse_jacobi_cn_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T inverse_jacobi_elliptic_cn(T x, T m) {
    const T tol = detail::inverse_jacobi_cn_tolerance<T>();

    // Handle special case x = 1: arccn(1, m) = 0
    if (std::abs(x - T(1)) < tol) {
        return T(0);
    }

    // Handle special case m = 0: arccn(x, 0) = arccos(x)
    if (std::abs(m) < tol) {
        return std::acos(x);
    }

    // Handle special case m = 1: arccn(x, 1) = arcsech(x) = acosh(1/x)
    // Note: sech(u) = 1/cosh(u), so arcsech(x) = acosh(1/x)
    if (std::abs(m - T(1)) < tol) {
        // arcsech(x) = log((1 + sqrt(1 - x^2)) / x) for 0 < x <= 1
        T sqrt_term = std::sqrt(T(1) - x * x);
        return std::log((T(1) + sqrt_term) / x);
    }

    // General case: arccn(x, m) = sqrt(1 - x^2) * R_F(x^2, 1 - m + m*x^2, 1)
    T x2 = x * x;
    T one_minus_x2 = T(1) - x2;
    T sqrt_term = std::sqrt(one_minus_x2);

    T arg1 = x2;
    T arg2 = T(1) - m + m * x2;  // = 1 - m*(1 - x^2) = 1 - m + m*x^2
    T arg3 = T(1);

    return sqrt_term * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

template <typename T>
c10::complex<T> inverse_jacobi_elliptic_cn(c10::complex<T> x, c10::complex<T> m) {
    const T tol = detail::inverse_jacobi_cn_tolerance<T>();
    c10::complex<T> one(T(1), T(0));
    c10::complex<T> zero(T(0), T(0));

    // Handle special case x = 1: arccn(1, m) = 0
    if (std::abs(x - one) < tol) {
        return zero;
    }

    // Handle special case m = 0: arccn(x, 0) = arccos(x)
    if (std::abs(m) < tol) {
        return std::acos(x);
    }

    // Handle special case m = 1: arccn(x, 1) = arcsech(x)
    if (std::abs(m - one) < tol) {
        c10::complex<T> sqrt_term = std::sqrt(one - x * x);
        return std::log((one + sqrt_term) / x);
    }

    // General case: arccn(x, m) = sqrt(1 - x^2) * R_F(x^2, 1 - m + m*x^2, 1)
    c10::complex<T> x2 = x * x;
    c10::complex<T> one_minus_x2 = one - x2;
    c10::complex<T> sqrt_term = std::sqrt(one_minus_x2);

    c10::complex<T> arg1 = x2;
    c10::complex<T> arg2 = one - m + m * x2;
    c10::complex<T> arg3 = one;

    return sqrt_term * carlson_elliptic_integral_r_f(arg1, arg2, arg3);
}

} // namespace torchscience::kernel::special_functions
