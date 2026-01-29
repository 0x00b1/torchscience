#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "bessel_j.h"
#include "spherical_bessel_j_0.h"
#include "spherical_bessel_j_1.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for spherical_bessel_j
template <typename T>
constexpr T spherical_bessel_j_eps();

template <>
constexpr float spherical_bessel_j_eps<float>() { return 1e-6f; }

template <>
constexpr double spherical_bessel_j_eps<double>() { return 1e-12; }

template <>
inline c10::Half spherical_bessel_j_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 spherical_bessel_j_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

// Forward recurrence for spherical Bessel functions
// j_{n+1}(z) = (2n+1)/z * j_n(z) - j_{n-1}(z)
// This is stable for z > n
template <typename T>
T spherical_bessel_j_forward_recurrence(int n_int, T z) {
    if (n_int == 0) return spherical_bessel_j_0(z);
    if (n_int == 1) return spherical_bessel_j_1(z);

    T j_prev = spherical_bessel_j_0(z);
    T j_curr = spherical_bessel_j_1(z);

    for (int k = 1; k < n_int; ++k) {
        T j_next = (T(2 * k + 1) / z) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }

    return j_curr;
}

// Backward recurrence for spherical Bessel functions (Miller's algorithm)
// More stable for z < n
template <typename T>
T spherical_bessel_j_backward_recurrence(int n_int, T z) {
    // Start from large m and recur downward
    // Use normalization with j_0
    int m = n_int + static_cast<int>(std::sqrt(T(40) * std::max(T(n_int), static_cast<T>(std::abs(z)))) + std::abs(z));
    if (m < n_int + 20) m = n_int + 20;

    T j_next = T(0);
    T j_curr = T(1);
    T j_n = T(0);

    for (int k = m; k >= 0; --k) {
        if (k == n_int) {
            j_n = j_curr;
        }

        if (k == 0) break;

        T j_prev = (T(2 * k + 1) / z) * j_curr - j_next;
        j_next = j_curr;
        j_curr = j_prev;
    }

    // Normalize using j_0
    T j0_computed = j_curr;
    T j0_actual = spherical_bessel_j_0(z);

    if (std::abs(j0_computed) < spherical_bessel_j_eps<T>()) {
        // Fall back to computing via Bessel J
        const T pi = static_cast<T>(M_PI);
        T nu = T(n_int) + T(0.5);
        return std::sqrt(pi / (T(2) * z)) * bessel_j(nu, z);
    }

    return j_n * j0_actual / j0_computed;
}

// Complex forward recurrence
template <typename T>
c10::complex<T> spherical_bessel_j_forward_recurrence(int n_int, c10::complex<T> z) {
    if (n_int == 0) return spherical_bessel_j_0(z);
    if (n_int == 1) return spherical_bessel_j_1(z);

    c10::complex<T> j_prev = spherical_bessel_j_0(z);
    c10::complex<T> j_curr = spherical_bessel_j_1(z);

    for (int k = 1; k < n_int; ++k) {
        c10::complex<T> factor(T(2 * k + 1), T(0));
        c10::complex<T> j_next = (factor / z) * j_curr - j_prev;
        j_prev = j_curr;
        j_curr = j_next;
    }

    return j_curr;
}

// Complex backward recurrence
template <typename T>
c10::complex<T> spherical_bessel_j_backward_recurrence(int n_int, c10::complex<T> z) {
    T z_mag = std::abs(z);
    int m = n_int + static_cast<int>(std::sqrt(T(40) * std::max(T(n_int), z_mag)) + z_mag);
    if (m < n_int + 20) m = n_int + 20;

    c10::complex<T> j_next(T(0), T(0));
    c10::complex<T> j_curr(T(1), T(0));
    c10::complex<T> j_n(T(0), T(0));

    for (int k = m; k >= 0; --k) {
        if (k == n_int) {
            j_n = j_curr;
        }

        if (k == 0) break;

        c10::complex<T> factor(T(2 * k + 1), T(0));
        c10::complex<T> j_prev = (factor / z) * j_curr - j_next;
        j_next = j_curr;
        j_curr = j_prev;
    }

    // Normalize using j_0
    c10::complex<T> j0_computed = j_curr;
    c10::complex<T> j0_actual = spherical_bessel_j_0(z);

    if (std::abs(j0_computed) < spherical_bessel_j_eps<T>()) {
        // Fall back to computing via Bessel J
        const T pi = static_cast<T>(M_PI);
        c10::complex<T> nu(T(n_int) + T(0.5), T(0));
        c10::complex<T> two(T(2), T(0));
        c10::complex<T> pi_c(pi, T(0));
        return std::sqrt(pi_c / (two * z)) * bessel_j(nu, z);
    }

    return j_n * j0_actual / j0_computed;
}

} // namespace detail

// Spherical Bessel function of the first kind of general order n
// j_n(z) = sqrt(π/2z) * J_{n+1/2}(z)
template <typename T>
T spherical_bessel_j(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::spherical_bessel_j_eps<T>();

    // Check if n is a non-negative integer
    T n_rounded = std::round(n);
    bool is_nonneg_integer = (n >= T(0)) && (std::abs(n - n_rounded) < eps);

    // For z = 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return T(1);  // j_0(0) = 1
        } else if (n > T(0)) {
            return T(0);  // j_n(0) = 0 for n > 0
        } else {
            // For n < 0, j_n(0) is singular for non-integer or diverges
            return std::numeric_limits<T>::infinity();
        }
    }

    // For integer n >= 0, use optimized implementations
    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Dispatch to optimized implementations for n=0, n=1
        if (n_int == 0) return spherical_bessel_j_0(z);
        if (n_int == 1) return spherical_bessel_j_1(z);

        // Choose recurrence direction based on |z| vs n
        if (std::abs(z) >= T(n_int)) {
            return detail::spherical_bessel_j_forward_recurrence(n_int, z);
        } else {
            return detail::spherical_bessel_j_backward_recurrence(n_int, z);
        }
    }

    // For negative integer n, use reflection formula
    // j_{-n-1}(z) = (-1)^{n+1} * y_n(z), but we compute via Bessel J
    // j_n(z) = sqrt(π/2z) * J_{n+1/2}(z)

    // General case: use the relation to Bessel J
    const T pi = static_cast<T>(M_PI);

    // Handle negative z
    if (z < T(0)) {
        // j_n(-z) = (-1)^n * j_n(z) for integer n
        // For non-integer n, use the general formula
        T nu = n + T(0.5);
        T prefix = std::sqrt(pi / (T(2) * std::abs(z)));

        // For negative z, need to handle branch cut
        T j_val = bessel_j(nu, std::abs(z));

        // (-z)^nu = |z|^nu * e^{i*pi*nu} for principal branch
        // For real result with real n, use cos(pi*n)
        T phase = std::cos(pi * n);
        return phase * prefix * j_val;
    }

    T nu = n + T(0.5);
    T prefix = std::sqrt(pi / (T(2) * z));
    return prefix * bessel_j(nu, z);
}

// Complex version
template <typename T>
c10::complex<T> spherical_bessel_j(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::spherical_bessel_j_eps<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return one;  // j_0(0) = 1
        } else if (n.real() > T(0)) {
            return zero;  // j_n(0) = 0 for Re(n) > 0
        } else {
            return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
    }

    // Check if n is a real non-negative integer
    bool n_is_real = std::abs(n.imag()) < eps;
    T n_real = n.real();
    T n_rounded = std::round(n_real);
    bool is_nonneg_integer = n_is_real && (n_real >= T(0)) && (std::abs(n_real - n_rounded) < eps);

    // For non-negative integer n with real z, use optimized path
    bool z_is_real = std::abs(z.imag()) < eps;

    if (is_nonneg_integer) {
        int n_int = static_cast<int>(n_rounded);

        if (n_int == 0) return spherical_bessel_j_0(z);
        if (n_int == 1) return spherical_bessel_j_1(z);

        T z_mag = std::abs(z);
        if (z_mag >= T(n_int)) {
            return detail::spherical_bessel_j_forward_recurrence(n_int, z);
        } else {
            return detail::spherical_bessel_j_backward_recurrence(n_int, z);
        }
    }

    // General case: use the relation to Bessel J
    // j_n(z) = sqrt(π/2z) * J_{n+1/2}(z)
    const T pi = static_cast<T>(M_PI);
    c10::complex<T> pi_c(pi, T(0));
    c10::complex<T> two(T(2), T(0));
    c10::complex<T> half(T(0.5), T(0));

    c10::complex<T> nu = n + half;
    c10::complex<T> prefix = std::sqrt(pi_c / (two * z));
    return prefix * bessel_j(nu, z);
}

} // namespace torchscience::kernel::special_functions
