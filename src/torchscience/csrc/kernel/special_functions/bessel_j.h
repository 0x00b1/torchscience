#pragma once

#include <cmath>
#include <limits>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>

#include "bessel_j_0.h"
#include "bessel_j_1.h"
#include "gamma.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Tolerance constants for bessel_j
template <typename T>
constexpr T bessel_j_eps();

template <>
constexpr float bessel_j_eps<float>() { return 1e-7f; }

template <>
constexpr double bessel_j_eps<double>() { return 1e-15; }

template <>
inline c10::Half bessel_j_eps<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 bessel_j_eps<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int bessel_j_max_iter() { return 300; }

// Power series for Jₙ(z):
// Jₙ(z) = (z/2)ⁿ * Σₖ₌₀^∞ (-z²/4)ᵏ / (k! * Γ(n+k+1))
template <typename T>
T bessel_j_series(T n, T z) {
    const T eps = bessel_j_eps<T>();
    const int max_iter = bessel_j_max_iter<T>();

    // For z=0: J_0(0) = 1, J_n(0) = 0 for n > 0
    if (z == T(0)) {
        if (n == T(0)) {
            return T(1);
        } else if (n > T(0)) {
            return T(0);
        } else {
            // For n < 0, need to be careful
            // J_{-n}(0) = (-1)^n * J_n(0) for integer n
            T n_rounded = std::round(n);
            if (std::abs(n - n_rounded) < eps) {
                int n_int = static_cast<int>(n_rounded);
                if (n_int == 0) return T(1);
                return T(0);  // J_n(0) = 0 for n != 0 integer
            }
            // For non-integer negative n, J_n(0) is singular (involves 0^n with n < 0)
            return std::numeric_limits<T>::infinity();
        }
    }

    // Compute (z/2)^n
    T z_half = z / T(2);
    T prefix;
    if (n == T(0)) {
        prefix = T(1);
    } else {
        prefix = std::pow(z_half, n);
    }

    // Handle potential overflow/underflow in prefix
    if (!std::isfinite(prefix)) {
        if (prefix == T(0)) return T(0);
        return prefix;  // inf or nan
    }

    // Compute the series: Σₖ₌₀^∞ (-z²/4)ᵏ / (k! * Γ(n+k+1))
    T z2_over_4 = -(z * z) / T(4);
    T term = T(1) / gamma(n + T(1));
    T sum = term;

    for (int k = 1; k <= max_iter; ++k) {
        // term_k = term_{k-1} * (-z²/4) / (k * (n+k))
        term *= z2_over_4 / (T(k) * (n + T(k)));
        sum += term;

        if (std::abs(term) < eps * std::abs(sum)) {
            break;
        }
    }

    return prefix * sum;
}

// Complex power series for Jₙ(z)
template <typename T>
c10::complex<T> bessel_j_series(c10::complex<T> n, c10::complex<T> z) {
    const T eps = bessel_j_eps<T>();
    const int max_iter = bessel_j_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z=0
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
    c10::complex<T> z2_over_4 = -(z * z) / c10::complex<T>(T(4), T(0));
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

// Miller's backward recurrence for computing Jₙ for integer n
// This is more stable than forward recurrence for large n
template <typename T>
T bessel_j_miller(int n_int, T z) {
    if (n_int < 0) {
        // Use J_{-n}(z) = (-1)^n * J_n(z)
        T result = bessel_j_miller(-n_int, z);
        return ((-n_int) % 2 == 0) ? result : -result;
    }

    if (n_int == 0) return bessel_j_0(z);
    if (n_int == 1) return bessel_j_1(z);

    // For small z, series is better
    if (std::abs(z) < T(1)) {
        return bessel_j_series(T(n_int), z);
    }

    // Miller's algorithm: start from large m and recur downward
    // Recurrence: J_{k-1}(z) = (2k/z) * J_k(z) - J_{k+1}(z)
    // Normalize using J_0

    // The starting index m should be large enough that J_m(z) is negligible
    // A good heuristic is m >= n + sqrt(40 * max(n, z)) + z
    // This ensures convergence for both large n and large z
    T nz_max = std::max(T(n_int), z);
    int m = n_int + static_cast<int>(std::sqrt(T(40) * nz_max) + z);
    if (m < n_int + 20) m = n_int + 20;

    // j_curr represents J_k, j_next represents J_{k+1}
    T j_next = T(0);  // J_{m+1} = 0 (arbitrary starting point)
    T j_curr = T(1);  // J_m = 1 (will be normalized later)
    T j_n = T(0);     // Will store J_{n_int}

    for (int k = m; k >= 0; --k) {
        if (k == n_int) {
            // Save the unnormalized J_n before we overwrite j_curr
            j_n = j_curr;
        }

        if (k == 0) {
            // We have j_curr = J_0 (unnormalized), done with recurrence
            break;
        }

        // Compute J_{k-1} = (2k/z) * J_k - J_{k+1}
        T j_prev = (T(2 * k) / z) * j_curr - j_next;
        j_next = j_curr;
        j_curr = j_prev;
    }

    // j_curr is now proportional to J_0
    // Normalize: J_0(z) is known
    T j0_computed = j_curr;
    T j0_actual = bessel_j_0(z);

    if (std::abs(j0_computed) < bessel_j_eps<T>()) {
        // j0_computed is too small; fall back to series
        return bessel_j_series(T(n_int), z);
    }

    return j_n * j0_actual / j0_computed;
}

// Asymptotic expansion for large |z|
// Jₙ(z) ~ sqrt(2/(πz)) * [P(n,z)*cos(χ) - Q(n,z)*sin(χ)]
// where χ = z - (n/2 + 1/4)*π
template <typename T>
T bessel_j_asymptotic(T n, T z) {
    const T pi = static_cast<T>(M_PI);

    T chi = z - (n / T(2) + T(0.25)) * pi;

    // P and Q series (first few terms)
    T mu = T(4) * n * n;
    T inv_8z = T(1) / (T(8) * z);

    T p = T(1);
    T q = (mu - T(1)) * inv_8z;

    // Add more terms if needed for accuracy
    T term_p = T(1);
    T term_q = (mu - T(1)) * inv_8z;

    // Second order terms
    T factor = inv_8z * inv_8z;
    term_p = -((mu - T(1)) * (mu - T(9))) * factor / T(2);
    p += term_p;

    term_q = ((mu - T(1)) * (mu - T(9)) * (mu - T(25))) * factor * inv_8z / T(6);
    q += term_q;

    T amplitude = std::sqrt(T(2) / (pi * z));
    return amplitude * (p * std::cos(chi) - q * std::sin(chi));
}

} // namespace detail

template <typename T>
T bessel_j(T n, T z) {
    // Handle special values
    if (std::isnan(n) || std::isnan(z)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    const T eps = detail::bessel_j_eps<T>();

    // Check if n is an integer
    T n_rounded = std::round(n);
    bool is_integer = std::abs(n - n_rounded) < eps;

    // For z = 0
    if (z == T(0)) {
        if (std::abs(n) < eps) {
            return T(1);  // J_0(0) = 1
        } else if (n > T(0)) {
            return T(0);  // J_n(0) = 0 for n > 0
        } else if (is_integer) {
            // J_{-n}(0) = (-1)^n * J_n(0) = 0 for n != 0
            return T(0);
        } else {
            // Non-integer negative n at z=0: singular
            return std::numeric_limits<T>::infinity();
        }
    }

    // Handle negative z for real n
    // J_n(-z) = (-1)^n * J_n(z) for integer n
    // J_n(-z) = e^{i*n*π} * J_n(z) for non-integer n (complex result, but we're real-only here)
    if (z < T(0)) {
        if (is_integer) {
            int n_int = static_cast<int>(n_rounded);
            T result = bessel_j(n, -z);
            return (n_int % 2 == 0) ? result : -result;
        } else {
            // For non-integer n, J_n(-z) is complex - return NaN for real implementation
            // Actually, we can handle it via the series for |z|
            // J_n(-z) = e^{i*n*π} * J_n(z)
            // For real n, this gives cos(n*π) * J_n(|z|) (ignoring imaginary part)
            // This is an approximation - for full correctness, use complex version
            T result = bessel_j(n, -z);
            return std::cos(static_cast<T>(M_PI) * n) * result;
        }
    }

    // For integer orders, use specialized methods
    if (is_integer) {
        int n_int = static_cast<int>(n_rounded);

        // Use direct formulas for n=0, n=1
        if (n_int == 0) return bessel_j_0(z);
        if (n_int == 1) return bessel_j_1(z);
        if (n_int == -1) return -bessel_j_1(z);

        // For |n| <= 20 or moderate z, use Miller's algorithm
        if (std::abs(n_int) <= 20 || z <= T(2) * std::abs(n_int)) {
            return detail::bessel_j_miller(n_int, z);
        }

        // For large z >> n, use asymptotic expansion
        if (z > T(30) + T(2) * std::abs(n_int)) {
            return detail::bessel_j_asymptotic(T(n_int), z);
        }

        // Default: Miller's algorithm
        return detail::bessel_j_miller(n_int, z);
    }

    // For non-integer orders, use series or asymptotic
    if (z <= T(20) + T(2) * std::abs(n)) {
        return detail::bessel_j_series(n, z);
    } else {
        return detail::bessel_j_asymptotic(n, z);
    }
}

// Complex version
template <typename T>
c10::complex<T> bessel_j(c10::complex<T> n, c10::complex<T> z) {
    const T eps = detail::bessel_j_eps<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> zero(T(0), T(0));

    // For z near 0
    if (std::abs(z) < eps) {
        if (std::abs(n) < eps) {
            return one;  // J_0(0) = 1
        } else if (n.real() > T(0) || (n.real() == T(0) && n.imag() != T(0))) {
            return zero;  // J_n(0) = 0 for Re(n) > 0
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
        T result = bessel_j(T(n_int), z.real());
        return c10::complex<T>(result, T(0));
    }

    // General case: use power series
    // For large |z|, asymptotic expansion would be better but is more complex for complex z
    return detail::bessel_j_series(n, z);
}

} // namespace torchscience::kernel::special_functions
