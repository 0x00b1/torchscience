#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "bessel_y.h"
#include "spherical_bessel_y_0.h"
#include "spherical_bessel_y_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for spherical_bessel_y
template <typename T>
constexpr T spherical_bessel_y_eps();

template <>
constexpr float spherical_bessel_y_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_y_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_y_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_y_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Forward recurrence for spherical Bessel functions of the second kind
// y_{n+1}(z) = (2n+1)/z * y_n(z) - y_{n-1}(z)
// Forward recurrence is stable for y_n (unlike j_n)
template <typename T>
T spherical_bessel_y_forward_recurrence(int n_int, T z) {
    if (n_int == 0) return spherical_bessel_y_0(z);
    if (n_int == 1) return spherical_bessel_y_1(z);

    T y_prev = spherical_bessel_y_0(z);
    T y_curr = spherical_bessel_y_1(z);

    for (int k = 1; k < n_int; ++k) {
        T y_next = (T(2 * k + 1) / z) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }

    return y_curr;
}

// Complex forward recurrence
template <typename T>
c10::complex<T> spherical_bessel_y_forward_recurrence(int n_int, c10::complex<T> z) {
    if (n_int == 0) return spherical_bessel_y_0(z);
    if (n_int == 1) return spherical_bessel_y_1(z);

    c10::complex<T> y_prev = spherical_bessel_y_0(z);
    c10::complex<T> y_curr = spherical_bessel_y_1(z);

    for (int k = 1; k < n_int; ++k) {
        c10::complex<T> factor(T(2 * k + 1), T(0));
        c10::complex<T> y_next = (factor / z) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }

    return y_curr;
}

} // namespace detail

// Spherical Bessel function of the second kind of general order n
// y_n(z) = sqrt(pi/2z) * Y_{n+1/2}(z)
template <typename T>
T spherical_bessel_y(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::spherical_bessel_y_eps<T>();

    // Check if n is a non-negative integer
    T n_rounded = std::round(n);
    bool is_nonneg_integer = (n >= T(0)) && (std::abs(n - n_rounded) < eps);

    // For z = 0, y_n is singular for all n
    if (std::abs(z) < eps) {
        return -std::numeric_limits<T>::infinity();
    }

    // Handle negative z using symmetry: y_n(-z) = (-1)^(n+1) * y_n(z)
    if (z < T(0)) {
        if (is_nonneg_integer) {
            int n_int = static_cast<int>(n_rounded);
            T result = spherical_bessel_y(n, -z);
            // (-1)^(n+1) = -1 for even n, +1 for odd n
            return (n_int % 2 == 0) ? -result : result;
        }
        // For non-integer n with negative z, use the relation to Bessel Y
        const T pi = static_cast<T>(M_PI);
        T nu = n + T(0.5);
        T prefix = std::sqrt(pi / (T(2) * std::abs(z)));
        T y_val = bessel_y(nu, std::abs(z));
        // Phase factor for negative z
        T phase = std::cos(pi * (n + T(1)));
        return phase * prefix * y_val;
    }

    // For integer n >= 0, use optimized implementations
    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Dispatch to optimized implementations for n=0, n=1
        if (n_int == 0) return spherical_bessel_y_0(z);
        if (n_int == 1) return spherical_bessel_y_1(z);

        // Forward recurrence is stable for y_n
        return detail::spherical_bessel_y_forward_recurrence(n_int, z);
    }

    // General case: use the relation to Bessel Y
    // y_n(z) = sqrt(pi/2z) * Y_{n+1/2}(z)
    const T pi = static_cast<T>(M_PI);
    T nu = n + T(0.5);
    T prefix = std::sqrt(pi / (T(2) * z));
    return prefix * bessel_y(nu, z);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_y(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_y_eps<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0, y_n is singular
    if (std::abs(z) < eps) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    // Check if n is a real non-negative integer
    bool n_is_real = std::abs(n.imag()) < eps;
    T n_real = n.real();
    T n_rounded = std::round(n_real);
    bool is_nonneg_integer = n_is_real && (n_real >= T(0)) && (std::abs(n_real - n_rounded) < eps);

    // For non-negative integer n, use recurrence
    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        if (n_int == 0) return spherical_bessel_y_0(z);
        if (n_int == 1) return spherical_bessel_y_1(z);

        return detail::spherical_bessel_y_forward_recurrence(n_int, z);
    }

    // General case: use the relation to Bessel Y
    // y_n(z) = sqrt(pi/2z) * Y_{n+1/2}(z)
    const T pi = static_cast<T>(M_PI);
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> half(T(0.5), T(0));

    c10::complex<T> nu = n + half;
    c10::complex<T> prefix = std::sqrt(pi_c / (two * z));
    return prefix * bessel_y(nu, z);
}

} // namespace torchscience::kernel::special_functions
