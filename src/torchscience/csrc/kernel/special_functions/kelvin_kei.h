#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants for kei function
constexpr double KEI_SQRT2 = 1.4142135623730950488016887;  // sqrt(2)
constexpr double KEI_PI = 3.14159265358979323846264338;
constexpr double KEI_PIO4 = 0.78539816339744830961566085;  // pi/4
constexpr double KEI_PIO8 = 0.39269908169872415480783042;  // pi/8
constexpr double KEI_EULER_GAMMA = 0.5772156649015328606065121;  // Euler-Mascheroni constant

// Threshold for switching between power series and asymptotic expansion
constexpr double KEI_LARGE_X = 8.0;

} // namespace detail

// Kelvin function kei(x)
// kei(x) = Im[K_0(x * e^(i*pi/4))]
//
// For small x, we use the series expansion:
// kei(x) = -ln(x/2) * bei(x) - pi/4 * ber(x) + sum of correction series
//
// More specifically, using the expansion from Abramowitz & Stegun 9.9.12:
// kei(x) = -[ln(x/2) + gamma] * bei(x) - (pi/4) * ber(x)
//        + sum_{k=0}^inf [(-1)^k / ((2k+1)!)^2] * H_{2k+1} * (x/2)^(4k+2)
//
// where H_n = sum_{j=1}^n 1/j is the harmonic number.
//
// For large x, we use the asymptotic expansion:
// kei(x) ~ -sqrt(pi/(2x)) * exp(-x/sqrt(2)) * [f(x) * sin(alpha) - g(x) * cos(alpha)]
// where alpha = x/sqrt(2) + pi/8
//
template <typename T>
T kelvin_kei(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // kei is even function: kei(-x) = kei(x)
    x = std::abs(x);

    if (x == T(0)) {
        // kei(0) = -pi/4
        return T(-detail::KEI_PIO4);
    }

    if (std::isinf(x)) {
        // kei(x) -> 0 as x -> infinity (exponential decay)
        return T(0);
    }

    if (x <= T(detail::KEI_LARGE_X)) {
        // Power series for small to medium x
        // We need to compute ber(x), bei(x), and the correction series

        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        // Compute ber(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n) / ((2n)!)^2
        T ber_sum = T(1);
        T ber_term = T(1);

        // Compute bei(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n+2) / ((2n+1)!)^2
        T bei_sum = x2;  // First term (n=0)
        T bei_term = x2;

        // Compute the correction series with harmonic numbers
        // sum_{k=0}^inf [(-1)^k / ((2k+1)!)^2] * H_{2k+1} * (x/2)^(4k+2)
        // First term (k=0): H_1 * (x/2)^2 / (1!)^2 = 1 * x2
        T corr_sum = x2;  // H_1 = 1, factorial 1!^2 = 1
        T harmonic = T(1);  // H_1
        T factorial_odd_sq = T(1);  // (1!)^2

        for (int n = 1; n < 100; ++n) {
            // Update factorial: ((2n)!)^2 = ((2(n-1))!)^2 * ((2n-1) * (2n))^2
            T factor = T(2 * n - 1) * T(2 * n);
            T factorial_2n_sq_factor = factor * factor;

            // ber term
            ber_term = -ber_term * x4 / factorial_2n_sq_factor;
            ber_sum += ber_term;

            // bei term: bei uses (x/2)^(4n+2), so factor is different
            // bei_n / bei_{n-1} = -x^4 / ((2n)^2 * (2n+1)^2)
            T bei_factor = T(2 * n) * T(2 * n + 1);
            bei_term = -bei_term * x4 / (bei_factor * bei_factor);
            bei_sum += bei_term;

            // Update factorial for odd factorial: ((2n+1)!)^2 = ((2n-1)!)^2 * (2n)^2 * (2n+1)^2
            factorial_odd_sq *= T(2 * n) * T(2 * n) * T(2 * n + 1) * T(2 * n + 1);

            // Update harmonic number: H_{2n+1} = H_{2n-1} + 1/(2n) + 1/(2n+1)
            harmonic += T(1) / T(2 * n) + T(1) / T(2 * n + 1);

            // Correction series term: (-1)^n * H_{2n+1} * (x/2)^(4n+2) / ((2n+1)!)^2
            T sign = (n % 2 == 0) ? T(1) : T(-1);
            T x_power = x2;
            for (int j = 0; j < n; ++j) {
                x_power *= x4;
            }
            T corr_term = sign * harmonic * x_power / factorial_odd_sq;
            corr_sum += corr_term;

            // Check convergence
            if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
                std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
                std::abs(corr_term) < std::numeric_limits<T>::epsilon() * (std::abs(corr_sum) + T(1))) {
                break;
            }
        }

        // kei(x) = -(ln(x/2) + gamma) * bei(x) - (pi/4) * ber(x) + corr_sum
        T log_x_half = std::log(x_half);
        T gamma = T(detail::KEI_EULER_GAMMA);
        T pi_4 = T(detail::KEI_PIO4);

        return -(log_x_half + gamma) * bei_sum - pi_4 * ber_sum + corr_sum;
    } else {
        // Asymptotic expansion for large x
        // kei(x) ~ -sqrt(pi/(2x)) * exp(-x/sqrt(2)) * [f(x) * sin(alpha) - g(x) * cos(alpha)]
        // where alpha = x/sqrt(2) + pi/8
        //
        // From Abramowitz & Stegun 9.10.7-9.10.10:
        // The coefficients f and g are the same as for ker

        T sqrt2 = T(detail::KEI_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 + T(detail::KEI_PIO8);

        // Asymptotic coefficients (same as for ker)
        T inv_x = T(1) / x;
        T inv_x2 = inv_x * inv_x;
        T inv_x3 = inv_x2 * inv_x;
        T inv_x4 = inv_x2 * inv_x2;
        T inv_x5 = inv_x4 * inv_x;
        T inv_x6 = inv_x3 * inv_x3;

        // Coefficients from asymptotic expansion
        // f = 1 + f1/x + f2/x^2 + ...
        // g = g1/x + g2/x^2 + ...
        T f1 = T(-1) / (T(8) * sqrt2);        // -1/(8*sqrt(2))
        T g1 = T(-1) / (T(8) * sqrt2);        // -1/(8*sqrt(2))
        T f2 = T(1) / T(256);                 // 1/256
        T g2 = T(-1) / T(256);                // -1/256
        T f3 = T(25) / (T(3072) * sqrt2);     // 25/(3072*sqrt(2))
        T g3 = T(13) / (T(3072) * sqrt2);     // 13/(3072*sqrt(2))
        T f4 = T(-13) / T(16384);             // -13/16384
        T g4 = T(-13) / T(16384);             // -13/16384
        T f5 = T(-1073) / (T(196608) * sqrt2);
        T g5 = T(-697) / (T(196608) * sqrt2);
        T f6 = T(1033) / T(1048576);
        T g6 = T(-1033) / T(1048576);

        T f = T(1) + f1 * inv_x + f2 * inv_x2 + f3 * inv_x3 + f4 * inv_x4 + f5 * inv_x5 + f6 * inv_x6;
        T g = g1 * inv_x + g2 * inv_x2 + g3 * inv_x3 + g4 * inv_x4 + g5 * inv_x5 + g6 * inv_x6;

        T exp_factor = std::exp(-x_over_sqrt2);
        T sqrt_factor = std::sqrt(T(detail::KEI_PI) / (T(2) * x));

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        // kei uses -sin(alpha) for f and +cos(alpha) for g (opposite sign pattern from ker)
        return -exp_factor * sqrt_factor * (f * sin_alpha - g * cos_alpha);
    }
}

// Complex version
template <typename T>
c10::complex<T> kelvin_kei(c10::complex<T> x) {
    // For complex x, we use the power series which converges for all finite x
    // kei(x) = -(ln(x/2) + gamma) * bei(x) - (pi/4) * ber(x) + correction series

    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    // Compute ber(x)
    c10::complex<T> ber_sum(T(1), T(0));
    c10::complex<T> ber_term(T(1), T(0));

    // Compute bei(x)
    c10::complex<T> bei_sum = x2;
    c10::complex<T> bei_term = x2;

    // Compute correction series
    c10::complex<T> corr_sum = x2;  // H_1 = 1
    T harmonic = T(1);
    T factorial_odd_sq = T(1);

    for (int n = 1; n < 60; ++n) {
        T factor = T(2 * n - 1) * T(2 * n);
        T factorial_2n_sq_factor = factor * factor;

        // ber term
        ber_term = -ber_term * x4 / c10::complex<T>(factorial_2n_sq_factor, T(0));
        ber_sum = ber_sum + ber_term;

        // bei term
        T bei_factor = T(2 * n) * T(2 * n + 1);
        bei_term = -bei_term * x4 / c10::complex<T>(bei_factor * bei_factor, T(0));
        bei_sum = bei_sum + bei_term;

        // Update factorial and harmonic for correction series
        factorial_odd_sq *= T(2 * n) * T(2 * n) * T(2 * n + 1) * T(2 * n + 1);
        harmonic += T(1) / T(2 * n) + T(1) / T(2 * n + 1);

        // Correction term
        T sign = (n % 2 == 0) ? T(1) : T(-1);
        c10::complex<T> x_power = x2;
        for (int j = 0; j < n; ++j) {
            x_power = x_power * x4;
        }
        c10::complex<T> corr_term = c10::complex<T>(sign * harmonic / factorial_odd_sq, T(0)) * x_power;
        corr_sum = corr_sum + corr_term;

        if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
            std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum)) {
            break;
        }
    }

    c10::complex<T> log_x_half = std::log(x_half);
    T gamma = T(detail::KEI_EULER_GAMMA);
    T pi_4 = T(detail::KEI_PIO4);

    return -(log_x_half + c10::complex<T>(gamma, T(0))) * bei_sum
           - c10::complex<T>(pi_4, T(0)) * ber_sum + corr_sum;
}

} // namespace torchscience::kernel::special_functions
