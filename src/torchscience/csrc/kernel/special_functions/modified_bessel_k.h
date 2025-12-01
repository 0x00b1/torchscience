#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "modified_bessel_k_0.h"
#include "modified_bessel_k_1.h"
#include "modified_bessel_i_0.h"
#include "modified_bessel_i_1.h"
#include "gamma.h"
#include "digamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for modified_bessel_k
template <typename T>
constexpr T modified_bessel_k_eps();

template <>
constexpr float modified_bessel_k_eps<float>() { return 1e-7f; }

template <>
constexpr double modified_bessel_k_eps<double>() { return 1e-15; }

template <>
inline c10::Half modified_bessel_k_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 modified_bessel_k_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int modified_bessel_k_max_iter() { return 300; }

// Forward recurrence for K_n from K_0 and K_1 for integer n >= 2
// K_{n+1}(z) = K_{n-1}(z) + (2n/z) * K_n(z)
template <typename T>
T modified_bessel_k_forward_recurrence(int n_int, T z) {
    if (n_int == 0) return modified_bessel_k_0(z);
    if (n_int == 1) return modified_bessel_k_1(z);

    T k_prev = modified_bessel_k_0(z);  // K_0
    T k_curr = modified_bessel_k_1(z);  // K_1

    for (int k = 1; k < n_int; ++k) {
        // K_{k+1} = (2k/z) * K_k + K_{k-1}
        T k_next = (T(2 * k) / z) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }

    return k_curr;
}

// Complex forward recurrence
template <typename T>
c10::complex<T> modified_bessel_k_forward_recurrence_complex(int n_int, c10::complex<T> z) {
    if (n_int == 0) return modified_bessel_k_0(z);
    if (n_int == 1) return modified_bessel_k_1(z);

    c10::complex<T> k_prev = modified_bessel_k_0(z);  // K_0
    c10::complex<T> k_curr = modified_bessel_k_1(z);  // K_1

    for (int k = 1; k < n_int; ++k) {
        // K_{k+1} = (2k/z) * K_k + K_{k-1}
        c10::complex<T> k_next = (c10::complex<T>(T(2 * k), T(0)) / z) * k_curr + k_prev;
        k_prev = k_curr;
        k_curr = k_next;
    }

    return k_curr;
}

// Power series for K_n(z) using the connection formula and I_n(z)
// For non-integer n, we can use:
// K_n(z) = (pi/2) * [I_{-n}(z) - I_n(z)] / sin(n*pi)
//
// For the modified Bessel I_n(z):
// I_n(z) = (z/2)^n * sum_{k=0}^inf (z^2/4)^k / (k! * Gamma(n+k+1))
template <typename T>
T modified_bessel_i_series(T n, T z) {
    const T eps = modified_bessel_k_eps<T>();
    const int max_iter = modified_bessel_k_max_iter<T>();

    // For z=0: I_n(0) = 0 for n != 0, I_0(0) = 1
    if (z == T(0)) {
        if (std::abs(n) < eps) {
            return T(1);
        } else if (n > T(0)) {
            return T(0);
        } else {
            // For n < 0: I_{-n}(0) depends on whether -n is 0
            T neg_n = -n;
            if (neg_n < eps) return T(1);
            return T(0);
        }
    }

    // Compute (z/2)^n
    T z_half = z / T(2);
    T prefix;
    if (std::abs(n) < eps) {
        prefix = T(1);
    } else {
        prefix = std::pow(z_half, n);
    }

    if (!std::isfinite(prefix)) {
        if (prefix == T(0)) return T(0);
        return prefix;
    }

    // Compute the series: sum_{k=0}^inf (z^2/4)^k / (k! * Gamma(n+k+1))
    T z2_over_4 = (z * z) / T(4);
    T term = T(1) / gamma(n + T(1));
    T sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        term *= z2_over_4 / (T(k) * (n + T(k)));
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

// Connection formula for non-integer n:
// K_n(z) = (pi/2) * [I_{-n}(z) - I_n(z)] / sin(n*pi)
template <typename T>
T modified_bessel_k_connection_formula(T n, T z) {
    const T pi = static_cast<T>(M_PI);
    const T eps = modified_bessel_k_eps<T>();

    T sin_npi = std::sin(n * pi);

    // Check if sin(n*pi) is too small (n close to integer)
    if (std::abs(sin_npi) < eps) {
        // This shouldn't happen for truly non-integer n
        return std::numeric_limits<T>::quiet_NaN();
    }

    T i_neg_n = modified_bessel_i_series(-n, z);
    T i_n = modified_bessel_i_series(n, z);

    return (pi / T(2)) * (i_neg_n - i_n) / sin_npi;
}

// Complex power series for I_n(z)
template <typename T>
c10::complex<T> modified_bessel_i_series_complex(c10::complex<T> n, c10::complex<T> z) {
    const T eps = modified_bessel_k_eps<T>();
    const int max_iter = modified_bessel_k_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return one;
        } else if (n.real() > T(0)) {
            return zero;
        }
        return zero;
    }

    // Compute (z/2)^n
    c10::complex<T> z_half = z / c10::complex<T>(T(2), T(0));
    c10::complex<T> prefix = std::pow(z_half, n);

    if (!std::isfinite(std::abs(prefix))) {
        return prefix;
    }

    // Compute the series
    c10::complex<T> z2_over_4 = (z * z) / c10::complex<T>(T(4), T(0));
    c10::complex<T> term = one / gamma(n + one);
    c10::complex<T> sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        c10::complex<T> k_c(T(k), T(0));
        term *= z2_over_4 / (k_c * (n + k_c));
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

// Complex connection formula
template <typename T>
c10::complex<T> modified_bessel_k_connection_formula(c10::complex<T> n, c10::complex<T> z) {
    const c10::complex<T> pi_c(static_cast<T>(M_PI), T(0));
    const c10::complex<T> two(T(2), T(0));
    const T eps = modified_bessel_k_eps<T>();

    c10::complex<T> n_pi = n * pi_c;
    c10::complex<T> sin_npi = std::sin(n_pi);

    // Check if sin(n*pi) is too small
    if (std::abs(sin_npi) < eps) {
        return c10::complex<T>(std::numeric_limits<T>::quiet_NaN(), T(0));
    }

    c10::complex<T> i_neg_n = modified_bessel_i_series_complex(-n, z);
    c10::complex<T> i_n = modified_bessel_i_series_complex(n, z);

    return (pi_c / two) * (i_neg_n - i_n) / sin_npi;
}

// Asymptotic expansion for large |z|
// K_n(z) ~ sqrt(pi/(2z)) * exp(-z) * [1 + (4n^2 - 1)/(8z) + ...]
template <typename T>
T modified_bessel_k_asymptotic(T n, T z) {
    const T pi = static_cast<T>(M_PI);

    T mu = T(4) * n * n;
    T inv_8z = T(1) / (T(8) * z);

    // First few terms of the asymptotic expansion
    T series = T(1);
    T term = (mu - T(1)) * inv_8z;
    series += term;

    term *= (mu - T(9)) * inv_8z / T(2);
    series += term;

    term *= (mu - T(25)) * inv_8z / T(3);
    series += term;

    term *= (mu - T(49)) * inv_8z / T(4);
    series += term;

    T amplitude = std::sqrt(pi / (T(2) * z)) * std::exp(-z);
    return amplitude * series;
}

} // namespace detail

template <typename T>
T modified_bessel_k(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::modified_bessel_k_eps<T>();

    // K_n is only defined for z > 0 (for real z)
    if (z <= T(0)) {
        if (z == T(0)) {
            return std::numeric_limits<T>::infinity();
        }
        // For z < 0, K_n is complex-valued; return NaN for real implementation
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (std::isinf(z)) {
        return T(0);  // K_n(+inf) = 0 (exponential decay)
    }

    // K_{-n}(z) = K_n(z) (symmetric in n)
    T n_abs = std::abs(n);

    // Check if n is an integer
    T n_rounded = std::round(n_abs);
    bool is_integer = std::abs(n_abs - n_rounded) < eps;

    // For integer orders, use recurrence from K_0 and K_1
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Use direct formulas for n=0, n=1
        if (n_int == 0) return modified_bessel_k_0(z);
        if (n_int == 1) return modified_bessel_k_1(z);

        // For moderate n and z, use forward recurrence
        // Forward recurrence for K_n is stable (unlike I_n)
        if (n_int <= 50 || z > T(2) * n_int) {
            return detail::modified_bessel_k_forward_recurrence(n_int, z);
        }

        // For large z >> n, use asymptotic expansion
        if (z > T(30) + T(2) * std::abs(n_int)) {
            return detail::modified_bessel_k_asymptotic(T(n_int), z);
        }

        // Default: forward recurrence
        return detail::modified_bessel_k_forward_recurrence(n_int, z);
    }

    // For non-integer orders, use connection formula or asymptotic
    if (z <= T(20) + T(2) * std::abs(n_abs)) {
        return detail::modified_bessel_k_connection_formula(n_abs, z);
    } else {
        return detail::modified_bessel_k_asymptotic(n_abs, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> modified_bessel_k(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::modified_bessel_k_eps<T>();
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    T z_mag = std::abs(z);
    if (z_mag < eps) {
        return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
    }

    // Check if n is a real integer
    bool n_is_real = std::abs(n.imag()) < eps;
    T n_real = n.real();
    T n_abs = std::abs(n_real);
    T n_rounded = std::round(n_abs);
    bool is_integer = n_is_real && std::abs(n_abs - n_rounded) < eps;

    // Check if z is real and positive
    bool z_is_real_positive = std::abs(z.imag()) < eps && z.real() > T(0);

    // K_{-n}(z) = K_n(z), so we work with |n|
    c10::complex<T> n_use = n_is_real ? c10::complex<T>(n_abs, T(0)) : n;
    if (!n_is_real) {
        // For complex n, use |n| conceptually - but actually just use n directly
        // The symmetry K_{-n} = K_n still applies
        n_use = n;
    }

    // For real integer n and real positive z, use the real implementation
    if (is_integer && z_is_real_positive) {
        int n_int = static_cast<int>(n_rounded);
        T result = modified_bessel_k(T(n_int), z.real());
        return c10::complex<T>(result, T(0));
    }

    // For integer n with complex z, use forward recurrence
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);
        return detail::modified_bessel_k_forward_recurrence_complex(n_int, z);
    }

    // General case for non-integer n: use connection formula
    return detail::modified_bessel_k_connection_formula(n_use, z);
}

} // namespace torchscience::kernel::special_functions
