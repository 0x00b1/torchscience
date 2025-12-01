#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "modified_bessel_i_0.h"
#include "modified_bessel_i_1.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for modified_bessel_i
template <typename T>
constexpr T modified_bessel_i_eps();

template <>
constexpr float modified_bessel_i_eps<float>() { return 1e-7f; }

template <>
constexpr double modified_bessel_i_eps<double>() { return 1e-15; }

template <>
inline c10::Half modified_bessel_i_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 modified_bessel_i_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int modified_bessel_i_max_iter() { return 300; }

// Power series for Iₙ(z):
// Iₙ(z) = (z/2)ⁿ * Σₖ₌₀^∞ (z²/4)ᵏ / (k! * Γ(n+k+1))
// Named with _general suffix to avoid conflict with modified_bessel_k.h
template <typename T>
T modified_bessel_i_series_general(T n, T z) {
    const T eps = modified_bessel_i_eps<T>();
    const int max_iter = modified_bessel_i_max_iter<T>();

    // For z=0: I_n(0) = 0 for n != 0, I_0(0) = 1
    if (z == T(0)) {
        if (std::abs(n) < eps) {
            return T(1);
        } else if (n > T(0)) {
            return T(0);
        } else {
            // For negative n: I_{-n}(0) depends on whether -n is an integer
            // I_{-n}(z) = I_n(z) for integer n
            T neg_n = -n;
            T n_rounded = std::round(neg_n);
            if (std::abs(neg_n - n_rounded) < eps) {
                // Integer case: I_{-n}(0) = I_n(0) = 0 for n != 0
                if (std::abs(n_rounded) < eps) return T(1);
                return T(0);
            }
            // Non-integer negative n at z=0: (z/2)^n term with n<0 is singular
            return std::numeric_limits<T>::infinity();
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

    // Handle potential overflow/underflow in prefix
    if (!std::isfinite(prefix)) {
        if (prefix == T(0)) return T(0);
        return prefix;  // inf or nan
    }

    // Compute the series: Σₖ₌₀^∞ (z²/4)ᵏ / (k! * Γ(n+k+1))
    T z2_over_4 = (z * z) / T(4);
    T term = T(1) / gamma(n + T(1));
    T sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        // term_k = term_{k-1} * (z²/4) / (k * (n+k))
        term *= z2_over_4 / (T(k) * (n + T(k)));
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

// Complex power series for Iₙ(z)
// Named with _general suffix to avoid conflict with modified_bessel_k.h
template <typename T>
c10::complex<T> modified_bessel_i_series_general(c10::complex<T> n, c10::complex<T> z) {
    const T eps = modified_bessel_i_eps<T>();
    const int max_iter = modified_bessel_i_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return one;
        } else if (n.real() > T(0)) {
            return zero;
        } else {
            return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
    }

    // Compute (z/2)^n
    c10::complex<T> z_half = z / c10::complex<T>(T(2), T(0));
    c10::complex<T> prefix = std::pow(z_half, n);

    // Handle potential overflow/underflow
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

// Forward recurrence for I_n from I_0 and I_1 for integer n
// Recurrence: I_{n+1}(z) = I_{n-1}(z) - (2n/z) * I_n(z)
// Note: Forward recurrence for I_n is unstable for large n, but we use Miller's backward recurrence for that
template <typename T>
T modified_bessel_i_forward_recurrence(int n_int, T z) {
    if (n_int < 0) {
        // Use I_{-n}(z) = I_n(z) for integer n
        return modified_bessel_i_forward_recurrence(-n_int, z);
    }

    if (n_int == 0) return modified_bessel_i_0(z);
    if (n_int == 1) return modified_bessel_i_1(z);

    // For small z or small n, forward recurrence can work
    // But we prefer Miller's backward recurrence for stability
    return modified_bessel_i_series_general(T(n_int), z);
}

// Miller's backward recurrence for computing Iₙ for integer n
// This is more stable than forward recurrence for large n
template <typename T>
T modified_bessel_i_miller(int n_int, T z) {
    if (n_int < 0) {
        // Use I_{-n}(z) = I_n(z) for integer n
        return modified_bessel_i_miller(-n_int, z);
    }

    if (n_int == 0) return modified_bessel_i_0(z);
    if (n_int == 1) return modified_bessel_i_1(z);

    // For small z, series is better
    if (std::abs(z) < T(1)) {
        return modified_bessel_i_series_general(T(n_int), z);
    }

    // For small n relative to z, forward recurrence might be OK
    // But for large n, we need Miller's backward recurrence

    // Miller's algorithm: start from large m and recur downward
    // Recurrence (backward): I_{k-1}(z) = (2k/z) * I_k(z) + I_{k+1}(z)
    // This is the inverse of: I_{k+1}(z) = I_{k-1}(z) - (2k/z) * I_k(z)
    // Normalize using I_0

    // The starting index m should be large enough
    T nz_max = std::max(T(n_int), z);
    int m = n_int + static_cast<int>(std::sqrt(T(40) * nz_max) + z);
    if (m < n_int + 20) m = n_int + 20;

    // i_curr represents I_k, i_next represents I_{k+1}
    T i_next = T(0);  // I_{m+1} = 0 (arbitrary starting point)
    T i_curr = T(1);  // I_m = 1 (will be normalized later)
    T i_n = T(0);     // Will store I_{n_int}

    for (int k = m; k >= 0; --k) {
        if (k == n_int) {
            // Save the unnormalized I_n before we overwrite i_curr
            i_n = i_curr;
        }

        if (k == 0) {
            // We have i_curr = I_0 (unnormalized), done with recurrence
            break;
        }

        // Compute I_{k-1} = (2k/z) * I_k + I_{k+1}
        T i_prev = (T(2 * k) / z) * i_curr + i_next;
        i_next = i_curr;
        i_curr = i_prev;
    }

    // i_curr is now proportional to I_0
    // Normalize: I_0(z) is known
    T i0_computed = i_curr;
    T i0_actual = modified_bessel_i_0(z);

    if (std::abs(i0_computed) < modified_bessel_i_eps<T>()) {
        // i0_computed is too small; fall back to series
        return modified_bessel_i_series_general(T(n_int), z);
    }

    return i_n * i0_actual / i0_computed;
}

// Complex forward recurrence
template <typename T>
c10::complex<T> modified_bessel_i_forward_recurrence_complex(int n_int, c10::complex<T> z) {
    if (n_int < 0) {
        return modified_bessel_i_forward_recurrence_complex(-n_int, z);
    }

    if (n_int == 0) return modified_bessel_i_0(z);
    if (n_int == 1) return modified_bessel_i_1(z);

    // Use series for complex case
    return modified_bessel_i_series_general(c10::complex<T>(T(n_int), T(0)), z);
}

// Asymptotic expansion for large |z|
// Iₙ(z) ~ exp(z) / sqrt(2πz) * [1 - (4n² - 1)/(8z) + ...]
template <typename T>
T modified_bessel_i_asymptotic(T n, T z) {
    const T pi = static_cast<T>(M_PI);

    T mu = T(4) * n * n;
    T inv_8z = T(1) / (T(8) * z);

    // First few terms of the asymptotic expansion
    T series = T(1);
    T term = -(mu - T(1)) * inv_8z;
    series += term;

    term *= -(mu - T(9)) * inv_8z / T(2);
    series += term;

    term *= -(mu - T(25)) * inv_8z / T(3);
    series += term;

    term *= -(mu - T(49)) * inv_8z / T(4);
    series += term;

    T amplitude = std::exp(z) / std::sqrt(T(2) * pi * z);
    return amplitude * series;
}

} // namespace detail

template <typename T>
T modified_bessel_i(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::modified_bessel_i_eps<T>();

    // Check if n is an integer
    T n_rounded = std::round(n);
    bool is_integer = std::abs(n - n_rounded) < eps;

    // For z = 0
    if (z == T(0)) {
        if (std::abs(n) < eps) {
            return T(1);  // I_0(0) = 1
        } else if (is_integer) {
            // I_n(0) = 0 for n != 0 integer (and I_{-n}(0) = I_n(0))
            return T(0);
        } else if (n > T(0)) {
            return T(0);  // I_n(0) = 0 for n > 0
        } else {
            // Non-integer negative n at z=0: singular (involves 0^n with n < 0)
            return std::numeric_limits<T>::infinity();
        }
    }

    if (std::isinf(z)) {
        if (z > T(0)) {
            return std::numeric_limits<T>::infinity();  // I_n(+inf) = +inf
        } else {
            // I_n(-inf): For integer n, I_n(-z) = (-1)^n * I_n(z)
            // So I_n(-inf) = +/- inf depending on n
            if (is_integer) {
                int n_int = static_cast<int>(n_rounded);
                if (n_int % 2 == 0) {
                    return std::numeric_limits<T>::infinity();
                } else {
                    return -std::numeric_limits<T>::infinity();
                }
            }
            return std::numeric_limits<T>::quiet_NaN();
        }
    }

    // Handle negative z for real n
    // I_n(-z) = (-1)^n * I_n(z) for integer n
    // I_n(-z) = e^{i*n*π} * I_n(z) for non-integer n (complex result)
    if (z < T(0)) {
        if (is_integer) {
            int n_int = static_cast<int>(n_rounded);
            T result = modified_bessel_i(n, -z);
            return (n_int % 2 == 0) ? result : -result;
        } else {
            // For non-integer n, I_n(-z) is complex
            // Return the real part: cos(n*π) * I_n(|z|)
            T result = modified_bessel_i(n, -z);
            return std::cos(static_cast<T>(M_PI) * n) * result;
        }
    }

    // I_{-n}(z) = I_n(z) for integer n
    if (is_integer && n < T(0)) {
        return modified_bessel_i(-n, z);
    }

    // For integer orders, use specialized methods
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Use direct formulas for n=0, n=1
        if (n_int == 0) return modified_bessel_i_0(z);
        if (n_int == 1) return modified_bessel_i_1(z);

        // For |n| <= 20 or moderate z, use Miller's algorithm
        if (std::abs(n_int) <= 20 || z <= T(2) * std::abs(n_int)) {
            return detail::modified_bessel_i_miller(n_int, z);
        }

        // For large z >> n, use asymptotic expansion
        if (z > T(30) + T(2) * std::abs(n_int)) {
            return detail::modified_bessel_i_asymptotic(T(n_int), z);
        }

        // Default: Miller's algorithm
        return detail::modified_bessel_i_miller(n_int, z);
    }

    // For non-integer orders, use series or asymptotic
    if (z <= T(20) + T(2) * std::abs(n)) {
        return detail::modified_bessel_i_series_general(n, z);
    } else {
        return detail::modified_bessel_i_asymptotic(n, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> modified_bessel_i(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::modified_bessel_i_eps<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return one;  // I_0(0) = 1
        } else if (n.real() > T(0) || (n.real() == T(0) && n.imag() != T(0))) {
            return zero;  // I_n(0) = 0 for Re(n) > 0
        } else {
            return c10::complex<T>(std::numeric_limits<T>::infinity(), T(0));
        }
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
        T result = modified_bessel_i(T(n_int), z.real());
        return c10::complex<T>(result, T(0));
    }

    // I_{-n}(z) = I_n(z) for integer n
    if (is_integer && n_real < T(0)) {
        c10::complex<T> n_pos(-n_real, T(0));
        return modified_bessel_i(n_pos, z);
    }

    // For integer n with complex z, use recurrence or series
    if (is_integer) {
        int n_int = static_cast<int>(std::abs(n_rounded));
        return detail::modified_bessel_i_forward_recurrence_complex(n_int, z);
    }

    // General case for non-integer n: use power series
    return detail::modified_bessel_i_series_general(n, z);
}

} // namespace torchscience::kernel::special_functions
