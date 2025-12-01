#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "bessel_j.h"
#include "bessel_y_0.h"
#include "bessel_y_1.h"
#include "gamma.h"
#include "digamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for bessel_y
template <typename T>
constexpr T bessel_y_eps();

template <>
constexpr float bessel_y_eps<float>() { return 1e-7f; }

template <>
constexpr double bessel_y_eps<double>() { return 1e-15; }

template <>
inline c10::Half bessel_y_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 bessel_y_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int bessel_y_max_iter() { return 300; }

// Forward recurrence for Y_n from Y_0 and Y_1 for integer n >= 2
// Y_{n+1}(z) = (2n/z) * Y_n(z) - Y_{n-1}(z)
template <typename T>
T bessel_y_forward_recurrence(int n_int, T z) {
    if (n_int == 0) return bessel_y_0(z);
    if (n_int == 1) return bessel_y_1(z);

    T y_prev = bessel_y_0(z);  // Y_0
    T y_curr = bessel_y_1(z);  // Y_1

    for (int k = 1; k < n_int; ++k) {
        // Y_{k+1} = (2k/z) * Y_k - Y_{k-1}
        T y_next = (T(2 * k) / z) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }

    return y_curr;
}

// Connection formula for non-integer n:
// Y_n(z) = [J_n(z) * cos(n*pi) - J_{-n}(z)] / sin(n*pi)
template <typename T>
T bessel_y_connection_formula(T n, T z) {
    const T pi = static_cast<T>(M_PI);

    T cos_npi = std::cos(n * pi);
    T sin_npi = std::sin(n * pi);

    // Check if sin(n*pi) is too small (n close to integer)
    if (std::abs(sin_npi) < bessel_y_eps<T>()) {
        // This shouldn't happen for truly non-integer n
        // Fall back to numerical limit approach
        return std::numeric_limits<T>::quiet_NaN();
    }

    T j_n = bessel_j(n, z);
    T j_neg_n = bessel_j(-n, z);

    return (j_n * cos_npi - j_neg_n) / sin_npi;
}

// Complex connection formula
template <typename T>
c10::complex<T> bessel_y_connection_formula(c10::complex<T> n, c10::complex<T> z) {
    const c10::complex<T> pi_c(static_cast<T>(M_PI), T(0));

    c10::complex<T> n_pi = n * pi_c;
    c10::complex<T> cos_npi = std::cos(n_pi);
    c10::complex<T> sin_npi = std::sin(n_pi);

    // Check if sin(n*pi) is too small
    if (std::abs(sin_npi) < bessel_y_eps<T>()) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
    }

    c10::complex<T> j_n = bessel_j(n, z);
    c10::complex<T> j_neg_n = bessel_j(-n, z);

    return (j_n * cos_npi - j_neg_n) / sin_npi;
}

// Asymptotic expansion for large |z|
// Y_n(z) ~ sqrt(2/(pi*z)) * [P(n,z)*sin(chi) + Q(n,z)*cos(chi)]
// where chi = z - (n/2 + 1/4)*pi
template <typename T>
T bessel_y_asymptotic(T n, T z) {
    const T pi = static_cast<T>(M_PI);

    T chi = z - (n / T(2) + T(0.25)) * pi;

    // P and Q series (first few terms)
    T mu = T(4) * n * n;
    T inv_8z = T(1) / (T(8) * z);

    T p = T(1);
    T q = (mu - T(1)) * inv_8z;

    // Second order terms
    T factor = inv_8z * inv_8z;
    T term_p = -((mu - T(1)) * (mu - T(9))) * factor / T(2);
    p += term_p;

    T term_q = ((mu - T(1)) * (mu - T(9)) * (mu - T(25))) * factor * inv_8z / T(6);
    q += term_q;

    T amplitude = std::sqrt(T(2) / (pi * z));
    // Y_n uses sin for P term and cos for Q term (opposite of J_n)
    return amplitude * (p * std::sin(chi) + q * std::cos(chi));
}

} // namespace detail

template <typename T>
T bessel_y(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::bessel_y_eps<T>();

    // Y_n has branch cut along negative real axis
    if (z <= T(0)) {
        if (z == T(0)) {
            return -std::numeric_limits<T>::infinity();
        }
        // For z < 0, Y_n is complex-valued; return NaN for real implementation
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::isinf(z)) {
        return T(0);  // Y_n(+inf) = 0 (oscillatory decay)
    }

    // Check if n is an integer
    T n_rounded = std::round(n);
    bool is_integer = std::abs(n - n_rounded) < eps;

    // For integer orders, use recurrence from Y_0 and Y_1
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Handle negative integer orders: Y_{-n}(z) = (-1)^n * Y_n(z)
        if (n_int < 0) {
            T result = bessel_y(T(-n_int), z);
            return ((-n_int) % 2 == 0) ? result : -result;
        }

        // Use direct formulas for n=0, n=1
        if (n_int == 0) return bessel_y_0(z);
        if (n_int == 1) return bessel_y_1(z);

        // For moderate n and z, use forward recurrence
        // Forward recurrence for Y_n is stable (unlike J_n)
        if (n_int <= 50 || z > T(2) * n_int) {
            return detail::bessel_y_forward_recurrence(n_int, z);
        }

        // For large z >> n, use asymptotic expansion
        if (z > T(30) + T(2) * std::abs(n_int)) {
            return detail::bessel_y_asymptotic(T(n_int), z);
        }

        // Default: forward recurrence
        return detail::bessel_y_forward_recurrence(n_int, z);
    }

    // For non-integer orders, use connection formula
    // Y_n(z) = [J_n(z) * cos(n*pi) - J_{-n}(z)] / sin(n*pi)
    if (z <= T(20) + T(2) * std::abs(n)) {
        return detail::bessel_y_connection_formula(n, z);
    } else {
        return detail::bessel_y_asymptotic(n, z);
    }
}

// Complex forward recurrence for Y_n from Y_0 and Y_1 for integer n >= 2
namespace detail {
template <typename T>
c10::complex<T> bessel_y_forward_recurrence_complex(int n_int, c10::complex<T> z) {
    if (n_int == 0) return bessel_y_0(z);
    if (n_int == 1) return bessel_y_1(z);

    c10::complex<T> y_prev = bessel_y_0(z);  // Y_0
    c10::complex<T> y_curr = bessel_y_1(z);  // Y_1

    for (int k = 1; k < n_int; ++k) {
        // Y_{k+1} = (2k/z) * Y_k - Y_{k-1}
        c10::complex<T> y_next = (c10::complex<T>(T(2 * k), T(0)) / z) * y_curr - y_prev;
        y_prev = y_curr;
        y_curr = y_next;
    }

    return y_curr;
}
} // namespace detail

// Complex version
template <typename T>
c10::complex<T> bessel_y(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::bessel_y_eps<T>();
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    T z_mag = std::abs(z);
    if (z_mag < eps) {
        return c10::complex<T>(-std::numeric_limits<T>::infinity(), T(0));
    }

    // Check if n is a real integer
    bool n_is_real = std::abs(n.imag()) < eps;
    T n_real = n.real();
    T n_rounded = std::round(n_real);
    bool is_integer = n_is_real && std::abs(n_real - n_rounded) < eps;

    // Check if z is real and positive
    bool z_is_real_positive = std::abs(z.imag()) < eps && z.real() > T(0);

    // For real integer n and real positive z, use the real implementation
    if (is_integer && z_is_real_positive) {
        int n_int = static_cast<int>(n_rounded);

        // Handle negative integer orders
        if (n_int < 0) {
            c10::complex<T> result = bessel_y(c10::complex<T>(T(-n_int), T(0)), z);
            return ((-n_int) % 2 == 0) ? result : -result;
        }

        // Use real implementation for efficiency
        T result = bessel_y(T(n_int), z.real());
        return c10::complex<T>(result, T(0));
    }

    // For integer n with complex z, use forward recurrence
    // (connection formula fails because sin(n*pi) = 0 for integer n)
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Handle negative integer orders: Y_{-n}(z) = (-1)^n * Y_n(z)
        if (n_int < 0) {
            c10::complex<T> result = bessel_y(c10::complex<T>(T(-n_int), T(0)), z);
            return ((-n_int) % 2 == 0) ? result : -result;
        }

        // Use forward recurrence for integer orders
        return detail::bessel_y_forward_recurrence_complex(n_int, z);
    }

    // General case for non-integer n: use connection formula
    // Y_n(z) = [J_n(z) * cos(n*pi) - J_{-n}(z)] / sin(n*pi)
    return detail::bessel_y_connection_formula(n, z);
}

} // namespace torchscience::kernel::special_functions
