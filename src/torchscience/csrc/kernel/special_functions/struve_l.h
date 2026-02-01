#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Convergence parameters for general order Struve L
template <typename T>
constexpr T struve_l_eps();

template <>
constexpr float struve_l_eps<float>() { return 1e-6f; }

template <>
constexpr double struve_l_eps<double>() { return 1e-14; }

template <>
inline c10::Half struve_l_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 struve_l_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int struve_l_max_iter() { return 300; }

// Power series for L_n(z):
// L_n(z) = (z/2)^(n+1) * sum_{k=0}^inf (z/2)^(2k) / [Gamma(k+3/2) * Gamma(k+n+3/2)]
//
// Note: Unlike H_n, there is no alternating sign (no (-1)^k)
//
// For k=0: term_0 = (z/2)^(n+1) / [Gamma(3/2) * Gamma(n+3/2)]
// Recurrence: term_{k+1} = term_k * (z/2)^2 / [(k+3/2) * (k+n+3/2)]
template <typename T>
T struve_l_series(T n, T z) {
    const T eps = struve_l_eps<T>();
    const int max_iter = struve_l_max_iter<T>();
    const T pi = static_cast<T>(M_PI);

    // Handle z = 0
    if (z == T(0)) {
        if (n >= T(-1)) {
            return T(0);
        } else {
            T n_plus_1 = n + T(1);
            T n_rounded = std::round(n_plus_1);
            if (std::abs(n_plus_1 - n_rounded) < eps && n_rounded <= T(0)) {
                return std::numeric_limits<T>::infinity();
            }
            return T(0);
        }
    }

    T z_half = z / T(2);
    T z_half_sq = z_half * z_half;

    // Compute (z/2)^(n+1)
    T prefix = std::pow(z_half, n + T(1));
    if (!std::isfinite(prefix)) {
        if (prefix == T(0)) return T(0);
        return prefix;
    }

    // Gamma(3/2) = sqrt(pi)/2
    T gamma_3_2 = std::sqrt(pi) / T(2);
    // Gamma(n+3/2)
    T gamma_n_3_2 = gamma(n + T(1.5));

    // First term: 1 / [Gamma(3/2) * Gamma(n+3/2)]
    T term = T(1) / (gamma_3_2 * gamma_n_3_2);
    T sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        // term_k = term_{k-1} * (z/2)^2 / [(k+1/2) * (k+n+1/2)]
        // Note: NO negative sign (unlike H_n)
        T factor_1 = T(k) + T(0.5);    // k + 1/2
        T factor_2 = T(k) + n + T(0.5); // k + n + 1/2
        T factor = z_half_sq / (factor_1 * factor_2);
        term *= factor;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

// Complex version
template <typename T>
c10::complex<T> struve_l_series(c10::complex<T> n, c10::complex<T> z) {
    const T eps = struve_l_eps<T>();
    const int max_iter = struve_l_max_iter<T>();
    const T pi = static_cast<T>(M_PI);
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // Handle z = 0
    if (std::abs(z) < eps) {
        if (n.real() >= T(-1)) {
            return zero;
        } else {
            return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
    }

    c10::complex<T> z_half = z / two;
    c10::complex<T> z_half_sq = z_half * z_half;

    // Compute (z/2)^(n+1)
    c10::complex<T> prefix = std::pow(z_half, n + one);
    if (!std::isfinite(std::abs(prefix))) {
        return prefix;
    }

    // Gamma(3/2) = sqrt(pi)/2
    T gamma_3_2 = std::sqrt(pi) / T(2);
    // Gamma(n+3/2)
    c10::complex<T> gamma_n_3_2 = gamma(n + c10::complex<T>(T(1.5), T(0)));

    // First term: 1 / [Gamma(3/2) * Gamma(n+3/2)]
    c10::complex<T> term = one / (c10::complex<T>(gamma_3_2, T(0)) * gamma_n_3_2);
    c10::complex<T> sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        c10::complex<T> k_c(T(k), T(0));
        c10::complex<T> factor_1 = k_c + half;      // k + 1/2
        c10::complex<T> factor_2 = k_c + n + half;  // k + n + 1/2
        // Note: NO negative sign (unlike H_n)
        c10::complex<T> factor = z_half_sq / (factor_1 * factor_2);
        term *= factor;
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

} // namespace detail

// Modified Struve function L_n(z) of general order n
//
// L_n(z) = (z/2)^(n+1) * sum_{k=0}^inf (z/2)^(2k) / [Gamma(k+3/2) * Gamma(k+n+3/2)]
//
// Relation to H_n: L_n(z) = -i * e^{-i*n*pi/2} * H_n(i*z)
//
// Properties:
// - L_n(-z) = (-1)^(n+1) * L_n(z) for integer n
// - L_n(0) = 0 for n >= -1
//
// Special cases:
// - n = 0: Use struve_l_0
// - n = 1: Use struve_l_1
template <typename T>
T struve_l(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::isinf(z)) {
        // L_n(+inf) = +inf for real z > 0
        // L_n(-inf) depends on n
        if (z > T(0)) {
            return std::numeric_limits<T>::infinity();
        } else {
            // For negative infinity, behavior depends on n
            return std::numeric_limits<T>::quiet_NaN();
        }
    }

    const T eps = detail::struve_l_eps<T>();

    // Handle z = 0
    if (z == T(0)) {
        if (n >= T(-1)) {
            return T(0);
        } else {
            T n_plus_1 = n + T(1);
            T n_rounded = std::round(n_plus_1);
            if (std::abs(n_plus_1 - n_rounded) < eps && n_rounded <= T(0)) {
                return std::numeric_limits<T>::infinity();
            }
            return T(0);
        }
    }

    // Handle negative z
    // L_n(-z) = (-1)^(n+1) * L_n(z) for integer n
    if (z < T(0)) {
        T n_rounded = std::round(n);
        bool is_integer = std::abs(n - n_rounded) < eps;

        if (is_integer) {
            int n_int = static_cast<int>(n_rounded);
            T sign = ((n_int + 1) % 2 == 0) ? T(1) : T(-1);
            return sign * struve_l(n, -z);
        } else {
            // For non-integer n, use the phase factor
            T result = struve_l(n, -z);
            return std::cos(static_cast<T>(M_PI) * (n + T(1))) * result;
        }
    }

    // Use power series
    return detail::struve_l_series(n, z);
}

// Complex version
template <typename T>
c10::complex<T> struve_l(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::struve_l_eps<T>();
    const c10::complex<T> zero(T(0), T(0));

    // Handle z = 0
    if (std::abs(z) < eps) {
        if (n.real() >= T(-1)) {
            return zero;
        } else {
            return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
    }

    // Use power series for all complex arguments
    return detail::struve_l_series(n, z);
}

} // namespace torchscience::kernel::special_functions
