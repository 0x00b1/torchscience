#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "modified_bessel_k.h"
#include "spherical_bessel_k_0.h"
#include "spherical_bessel_k_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for spherical_bessel_k
template <typename T>
constexpr T spherical_bessel_k_eps();

template <>
constexpr float spherical_bessel_k_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_k_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_k_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_k_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Forward recurrence for modified spherical Bessel functions of the second kind
// k_{n+1}(z) = k_{n-1}(z) + (2n+1)/z * k_n(z)
// Forward recurrence is stable for k_n (it grows with n)
template <typename T>
T spherical_bessel_k_forward_recurrence(int n_int, T z) {
    if (n_int == 0) return spherical_bessel_k_0(z);
    if (n_int == 1) return spherical_bessel_k_1(z);

    T k_prev = spherical_bessel_k_0(z);
    T k_curr = spherical_bessel_k_1(z);

    for (int k = 1; k < n_int; ++k) {
        // Recurrence: k_{k+1}(z) = (2k+1)/z * k_k(z) + k_{k-1}(z)
        T k_next = (T(2 * k + 1) / z) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }

    return k_curr;
}

// Complex forward recurrence
template <typename T>
c10::complex<T> spherical_bessel_k_forward_recurrence(int n_int, c10::complex<T> z) {
    if (n_int == 0) return spherical_bessel_k_0(z);
    if (n_int == 1) return spherical_bessel_k_1(z);

    c10::complex<T> k_prev = spherical_bessel_k_0(z);
    c10::complex<T> k_curr = spherical_bessel_k_1(z);

    for (int k = 1; k < n_int; ++k) {
        c10::complex<T> factor(T(2 * k + 1), T(0));
        c10::complex<T> k_next = (factor / z) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }

    return k_curr;
}

} // namespace detail

// Modified spherical Bessel function of the second kind of general order n
// k_n(z) = sqrt(pi/2z) * K_{n+1/2}(z)
template <typename T>
T spherical_bessel_k(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::spherical_bessel_k_eps<T>();

    // Check if n is a non-negative integer
    T n_rounded = std::round(n);
    bool is_nonneg_integer = (n >= T(0)) && (std::abs(n - n_rounded) < eps);

    // For z = 0, k_n is singular (pole at origin)
    if (std::abs(z) < eps) {
        return std::numeric_limits<T>::infinity();
    }

    // k_n(z) is only defined for z > 0 for real z
    // For z < 0, return NaN (complex-valued in general)
    if (z < T(0)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // For integer n >= 0, use optimized implementations
    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Dispatch to optimized implementations for n=0, n=1
        if (n_int == 0) return spherical_bessel_k_0(z);
        if (n_int == 1) return spherical_bessel_k_1(z);

        // Forward recurrence is stable for k_n
        return detail::spherical_bessel_k_forward_recurrence(n_int, z);
    }

    // General case: use the relation to modified Bessel K
    // k_n(z) = sqrt(pi/2z) * K_{n+1/2}(z)
    const T pi = static_cast<T>(M_PI);
    T nu = n + T(0.5);
    T prefix = std::sqrt(pi / (T(2) * z));
    return prefix * modified_bessel_k(nu, z);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_k(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_k_eps<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0, k_n is singular
    if (std::abs(z) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // Check if n is a real non-negative integer
    bool n_is_real = std::abs(n.imag()) < eps;
    T n_real = n.real();
    T n_rounded = std::round(n_real);
    bool is_nonneg_integer = n_is_real && (n_real >= T(0)) && (std::abs(n_real - n_rounded) < eps);

    // For non-negative integer n, use recurrence
    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        if (n_int == 0) return spherical_bessel_k_0(z);
        if (n_int == 1) return spherical_bessel_k_1(z);

        return detail::spherical_bessel_k_forward_recurrence(n_int, z);
    }

    // General case: use the relation to modified Bessel K
    // k_n(z) = sqrt(pi/2z) * K_{n+1/2}(z)
    const T pi = static_cast<T>(M_PI);
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> half(T(0.5), T(0));

    c10::complex<T> nu = n + half;
    c10::complex<T> prefix = std::sqrt(pi_c / (two * z));
    return prefix * modified_bessel_k(nu, z);
}

} // namespace torchscience::kernel::special_functions
