#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Mathematical constants for modified Struve functions
constexpr double STRUVE_L_TWO_OVER_PI = 0.6366197723675813430755350534900574;  // 2/pi
constexpr double STRUVE_L_PI = 3.14159265358979323846264338327950288;

// Convergence parameters
template <typename T>
inline T struve_l_zero_tolerance() {
    return T(1e-12);
}

template <>
inline float struve_l_zero_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_zero_tolerance<double>() { return 1e-12; }

template <typename T>
inline T struve_l_series_tolerance() {
    return std::numeric_limits<T>::epsilon() * T(10);
}

template <>
inline float struve_l_series_tolerance<float>() { return 1e-6f; }

template <>
inline double struve_l_series_tolerance<double>() { return 1e-14; }

// Modified Struve L_1 power series implementation
// L_1(z) = sum_{k=0}^inf (z/2)^(2k+2) / [Gamma(k+3/2) * Gamma(k+5/2)]
// Unlike H_1, there is NO (-1)^k alternating sign.
//
// Using recurrence for Gamma products:
// Gamma(k+3/2) = (k+1/2) * Gamma(k+1/2)
// Gamma(k+5/2) = (k+3/2) * Gamma(k+3/2)
//
// term_k = (z/2)^(2k+2) / [Gamma(k+3/2) * Gamma(k+5/2)]
// For k=0: Gamma(3/2)=sqrt(pi)/2, Gamma(5/2)=3*sqrt(pi)/4
//          Product = 3*pi/8
//          First term = (z/2)^2 / (3*pi/8) = (z/2)^2 * 8/(3*pi)
//
// Recurrence: term_{k+1} / term_k = (z/2)^2 / [(k+3/2)(k+5/2)]
// Note: No minus sign (unlike H_1 which has alternating signs)
template <typename T>
T struve_l_1_series(T z) {
    const T tolerance = struve_l_series_tolerance<T>();
    const int max_iterations = 200;

    T z_half = z / T(2);
    T z_half_sq = z_half * z_half;

    // First term (k=0): (z/2)^2 / [Gamma(3/2) * Gamma(5/2)]
    // Gamma(3/2) = sqrt(pi)/2, Gamma(5/2) = 3*sqrt(pi)/4
    // Gamma(3/2) * Gamma(5/2) = 3*pi/8
    // First term = (z/2)^2 / (3*pi/8) = (z/2)^2 * 8/(3*pi)
    T gamma_product = T(3) * T(STRUVE_L_PI) / T(8);  // Gamma(3/2) * Gamma(5/2)
    T term = z_half_sq / gamma_product;
    T sum = term;

    for (int k = 1; k <= max_iterations; ++k) {
        // term_{k} = term_{k-1} * (z/2)^2 / [(k+1/2)(k+3/2)]
        // Note: positive ratio (no alternating sign)
        T factor = z_half_sq / ((T(k) + T(0.5)) * (T(k) + T(1.5)));
        term *= factor;
        sum += term;

        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    return sum;
}

// Asymptotic expansion for L_1(z) for large |z|
// L_1(z) ~ I_1(z) - (2/pi) for large positive z
// For simplicity, we use the power series for all practical arguments
// since it converges well for |z| < 30
template <typename T>
T struve_l_1_asymptotic(T z) {
    // Fallback to series for large arguments (asymptotic not implemented)
    // This is acceptable since series converges (slowly) for all z
    return struve_l_1_series(z);
}

} // namespace detail

// Modified Struve function of the first kind, order 1
// L_1(z) = sum_{k=0}^inf (z/2)^(2k+2) / [Gamma(k+3/2) * Gamma(k+5/2)]
//
// Symmetry: L_1 is even: L_1(-z) = L_1(z)
// Special values: L_1(0) = 0
// For large positive z: L_1(z) ~ I_1(z) - 2/pi (grows exponentially)
template <typename T>
T struve_l_1(T z) {
    // Handle special values
    if (std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::isinf(z)) {
        // L_1(+inf) = +inf, L_1(-inf) = +inf (even function)
        return std::numeric_limits<T>::infinity();
    }

    // L_1 is even: L_1(-z) = L_1(z)
    T x = std::abs(z);

    if (x == T(0)) {
        return T(0);
    }

    // Use power series for small to moderate arguments
    // The series converges well for |z| < 20 or so
    if (x <= T(30)) {
        return detail::struve_l_1_series(x);
    }

    // Use asymptotic expansion for large arguments
    return detail::struve_l_1_asymptotic(x);
}

// Complex version of Modified Struve L_1
template <typename T>
c10::complex<T> struve_l_1(c10::complex<T> z) {
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
    const T tolerance = detail::struve_l_series_tolerance<T>();
    const int max_iterations = 200;

    Complex z_half = z / Complex(T(2), T(0));
    Complex z_half_sq = z_half * z_half;

    // First term (k=0): (z/2)^2 / (3*pi/8) = (z/2)^2 * 8/(3*pi)
    T gamma_product = T(3) * T(detail::STRUVE_L_PI) / T(8);
    Complex term = z_half_sq / Complex(gamma_product, T(0));
    Complex sum = term;

    for (int k = 1; k <= max_iterations; ++k) {
        // Note: positive ratio (no alternating sign)
        Complex factor = z_half_sq / Complex((T(k) + T(0.5)) * (T(k) + T(1.5)), T(0));
        term *= factor;
        sum += term;

        if (std::abs(term) < tolerance * std::abs(sum)) {
            break;
        }
    }

    return sum;
}

} // namespace torchscience::kernel::special_functions
