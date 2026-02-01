#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Mathematical constants for modified Struve functions
#ifndef TORCHSCIENCE_STRUVE_L_CONSTANTS_DEFINED
#define TORCHSCIENCE_STRUVE_L_CONSTANTS_DEFINED
constexpr double STRUVE_L0_TWO_OVER_PI = 0.6366197723675813430755350534900574;  // 2/pi
constexpr double STRUVE_L0_PI = 3.14159265358979323846264338327950288;
#endif

// Convergence parameters for L_0
template <typename T>
inline T struve_l_0_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_l_0_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_0_zero_tolerance<double>() { return 1e-12; }

template <typename T>
inline T struve_l_0_series_tolerance() {
    return std::numeric_limits<T>::epsilon() * T(10);
}

template <>
inline float struve_l_0_series_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_0_series_tolerance<double>() { return 1e-14; }

// Modified Struve L_0 power series implementation
// L_0(z) = sum_{k=0}^inf (z/2)^(2k+1) / [Gamma(k+3/2)]^2
// Unlike H_0, there is NO (-1)^k alternating sign.
//
// Using recurrence: Gamma(k+3/2) = (k+1/2) * Gamma(k+1/2)
//
// term_k = (z/2)^(2k+1) / [Gamma(k+3/2)]^2
// For k=0: Gamma(3/2) = sqrt(pi)/2, so [Gamma(3/2)]^2 = pi/4
// First term = (z/2) / (pi/4) = 2z/pi
//
// Recurrence: term_{k+1} / term_k = (z/2)^2 / (k+3/2)^2
// Note: No minus sign (unlike H_0 which has alternating signs)
template <typename T>
T struve_l_0_series(T z) {
    const T tolerance = struve_l_0_series_tolerance<T>();
    const int max_iterations = 200;

    T z_half = z / T(2);
    T z_half_sq = z_half * z_half;

    // First term (k=0): (z/2)^1 / [Gamma(3/2)]^2
    // Gamma(3/2) = sqrt(pi)/2
    // [Gamma(3/2)]^2 = pi/4
    // First term = (z/2) / (pi/4) = 2z/pi
    T gamma_sq = T(STRUVE_L0_PI) / T(4);  // [Gamma(3/2)]^2
    T term = z_half / gamma_sq;
    T sum = term;

    for (int k = 1; k <= max_iterations; ++k) {
        // term_{k} = term_{k-1} * (z/2)^2 / (k+1/2)^2
        // Note: positive ratio (no alternating sign)
        T factor_denom = T(k) + T(0.5);
        T factor = z_half_sq / (factor_denom * factor_denom);
        term *= factor;
        sum += term;

        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    return sum;
}

// Asymptotic expansion for L_0(z) for large |z|
// L_0(z) ~ I_0(z) - (2/pi) for large positive z
// For simplicity, we use the power series for all practical arguments
// since it converges well for |z| < 30
template <typename T>
T struve_l_0_asymptotic(T z) {
    // Fallback to series for large arguments (asymptotic not implemented)
    // This is acceptable since series converges (slowly) for all z
    return struve_l_0_series(z);
}

} // namespace detail

// Modified Struve function of the first kind, order 0
// L_0(z) = sum_{k=0}^inf (z/2)^(2k+1) / [Gamma(k+3/2)]^2
//
// Symmetry: L_0 is odd: L_0(-z) = -L_0(z)
// Special values: L_0(0) = 0
// For large positive z: L_0(z) ~ I_0(z) - 2/pi (grows exponentially)
template <typename T>
T struve_l_0(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::isinf(z)) {
        // L_0(+inf) = +inf, L_0(-inf) = -inf
        return z;
    }

    if (z == T(0)) {
        return T(0);
    }

    // L_0 is odd: L_0(-z) = -L_0(z)
    T sign = T(1);
    T x = z;
    if (z < T(0)) {
        x = -z;
        sign = T(-1);
    }

    // Use power series for small to moderate arguments
    if (x <= T(30)) {
        return sign * detail::struve_l_0_series(x);
    }

    // Use asymptotic expansion for large arguments
    return sign * detail::struve_l_0_asymptotic(x);
}

// Complex version of Modified Struve L_0
template <typename T>
c10::complex<T> struve_l_0(c10::complex<T> z) {
    using Complex = c10::complex<T>;

    T mag = std::abs(z);

    // Handle special cases
    if (std::isnan(z.real()) || std::isnan(z.imag())) {
        return Complex(std::numeric_limits<T>::quiet_NaN(),
                       std::numeric_limits<T>::quiet_NaN());
    }

    if (mag == T(0)) {
        return Complex(T(0), T(0));
    }

    // Power series for complex z
    const T tolerance = detail::struve_l_0_series_tolerance<T>();
    const int max_iterations = 200;

    Complex z_half = z / Complex(T(2), T(0));
    Complex z_half_sq = z_half * z_half;

    // First term (k=0): (z/2)^1 / [Gamma(3/2)]^2 = (z/2) / (pi/4) = 2z/pi
    T gamma_sq = T(detail::STRUVE_L0_PI) / T(4);
    Complex term = z_half / Complex(gamma_sq, T(0));
    Complex sum = term;

    for (int k = 1; k <= max_iterations; ++k) {
        T factor_denom = T(k) + T(0.5);
        // Note: positive ratio (no alternating sign)
        Complex factor = z_half_sq / Complex(factor_denom * factor_denom, T(0));
        term *= factor;
        sum += term;

        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    return sum;
}

} // namespace torchscience::kernel::special_functions
