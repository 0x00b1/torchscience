#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants for ker function
constexpr double KER_SQRT2 = 1.4142135623730950488016887;  // sqrt(2)
constexpr double KER_PI = 3.14159265358979323846264338;
constexpr double KER_PIO4 = 0.78539816339744830961566085;  // pi/4
constexpr double KER_PIO8 = 0.39269908169872415480783042;  // pi/8
constexpr double KER_EULER_GAMMA = 0.5772156649015328606065121;  // Euler-Mascheroni constant

// Threshold for switching between power series and asymptotic expansion
constexpr double KER_LARGE_X = 8.0;

} // namespace detail

// Kelvin function ker(x)
// ker(x) = Re[K_0(x * e^(i*pi/4))]
//
// For small x, we use the series expansion:
// ker(x) = -ln(x/2) * ber(x) + pi/4 * bei(x) + sum of correction series
//
// More specifically, using the expansion from Abramowitz & Stegun 9.9.10:
// ker(x) = -[ln(x/2) + gamma] * ber(x) + (pi/4) * bei(x)
//        + sum_{k=0}^inf [(-1)^k / ((2k)!)^2] * psi(2k+1) * (x/2)^(4k)
//
// where psi(n) = -gamma + sum_{j=1}^{n-1} 1/j is the digamma function evaluated at integer n.
//
// For large x, we use the asymptotic expansion:
// ker(x) ~ sqrt(pi/(2x)) * exp(-x/sqrt(2)) * [f(x) * cos(alpha) + g(x) * sin(alpha)]
// where alpha = x/sqrt(2) + pi/8
//
template <typename T>
T kelvin_ker(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // ker is even function: ker(-x) = ker(x)
    x = std::abs(x);

    if (x == T(0)) {
        // ker(0) = +infinity (logarithmic singularity)
        return std::numeric_limits<T>::infinity();
    }

    if (std::isinf(x)) {
        // ker(x) -> 0 as x -> infinity (exponential decay)
        return T(0);
    }

    if (x <= T(detail::KER_LARGE_X)) {
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

        // Compute the correction series with psi function
        // sum_{k=0}^inf [(-1)^k / ((2k)!)^2] * psi(2k+1) * (x/2)^(4k)
        // psi(1) = -gamma, psi(2) = -gamma + 1, psi(3) = -gamma + 1 + 1/2, ...
        // Actually for ker we need: sum_{k=0}^inf [(-1)^k / ((2k)!)^2] * H_{2k} * (x/2)^(4k)
        // where H_n = sum_{j=1}^n 1/j is the harmonic number (H_0 = 0)

        T corr_sum = T(0);  // H_0 = 0, so first term is 0
        T harmonic = T(0);  // H_0
        T factorial_sq = T(1);  // ((2k)!)^2

        for (int n = 1; n < 100; ++n) {
            // Update factorial: ((2n)!)^2 = ((2(n-1))!)^2 * ((2n-1) * (2n))^2
            T factor = T(2 * n - 1) * T(2 * n);
            factorial_sq *= factor * factor;

            // Update harmonic number: H_{2n} = H_{2(n-1)} + 1/(2n-1) + 1/(2n)
            harmonic += T(1) / T(2 * n - 1) + T(1) / T(2 * n);

            // ber term
            ber_term = -ber_term * x4 / (factor * factor);
            ber_sum += ber_term;

            // bei term: bei uses (x/2)^(4n+2), so factor is different
            // bei_n / bei_{n-1} = -x^4 / ((2n)^2 * (2n+1)^2)
            T bei_factor = T(2 * n) * T(2 * n + 1);
            bei_term = -bei_term * x4 / (bei_factor * bei_factor);
            bei_sum += bei_term;

            // Correction series term: (-1)^n * H_{2n} * (x/2)^(4n) / ((2n)!)^2
            T sign = (n % 2 == 0) ? T(1) : T(-1);
            T x_power = T(1);
            for (int j = 0; j < n; ++j) {
                x_power *= x4;
            }
            T corr_term = sign * harmonic * x_power / factorial_sq;
            corr_sum += corr_term;

            // Check convergence
            if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
                std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
                std::abs(corr_term) < std::numeric_limits<T>::epsilon() * (std::abs(corr_sum) + T(1))) {
                break;
            }
        }

        // ker(x) = -(ln(x/2) + gamma) * ber(x) + (pi/4) * bei(x) + corr_sum
        T log_x_half = std::log(x_half);
        T gamma = T(detail::KER_EULER_GAMMA);
        T pi_4 = T(detail::KER_PIO4);

        return -(log_x_half + gamma) * ber_sum + pi_4 * bei_sum + corr_sum;
    } else {
        // Asymptotic expansion for large x
        // ker(x) ~ sqrt(pi/(2x)) * exp(-x/sqrt(2)) * [f(x) * cos(alpha) + g(x) * sin(alpha)]
        // where alpha = x/sqrt(2) + pi/8
        //
        // From Abramowitz & Stegun 9.10.7-9.10.10:
        // f(x) = 1 + sum of correction terms
        // g(x) = sum of correction terms

        T sqrt2 = T(detail::KER_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 + T(detail::KER_PIO8);

        // Asymptotic coefficients for ker (from Abramowitz & Stegun)
        // These are the same as for modified Bessel K_0 with rotated argument
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
        T sqrt_factor = std::sqrt(T(detail::KER_PI) / (T(2) * x));

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        return exp_factor * sqrt_factor * (f * cos_alpha + g * sin_alpha);
    }
}

// Complex version
template <typename T>
c10::complex<T> kelvin_ker(c10::complex<T> x) {
    // For complex x, we use the power series which converges for all finite x
    // ker(x) = -(ln(x/2) + gamma) * ber(x) + (pi/4) * bei(x) + correction series

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
    c10::complex<T> corr_sum(T(0), T(0));
    T harmonic = T(0);
    T factorial_sq = T(1);

    for (int n = 1; n < 60; ++n) {
        T factor = T(2 * n - 1) * T(2 * n);
        factorial_sq *= factor * factor;
        harmonic += T(1) / T(2 * n - 1) + T(1) / T(2 * n);

        // ber term
        ber_term = -ber_term * x4 / c10::complex<T>(factor * factor, T(0));
        ber_sum = ber_sum + ber_term;

        // bei term
        T bei_factor = T(2 * n) * T(2 * n + 1);
        bei_term = -bei_term * x4 / c10::complex<T>(bei_factor * bei_factor, T(0));
        bei_sum = bei_sum + bei_term;

        // Correction term
        T sign = (n % 2 == 0) ? T(1) : T(-1);
        c10::complex<T> x_power(T(1), T(0));
        for (int j = 0; j < n; ++j) {
            x_power = x_power * x4;
        }
        c10::complex<T> corr_term = c10::complex<T>(sign * harmonic / factorial_sq, T(0)) * x_power;
        corr_sum = corr_sum + corr_term;

        if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
            std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum)) {
            break;
        }
    }

    c10::complex<T> log_x_half = std::log(x_half);
    T gamma = T(detail::KER_EULER_GAMMA);
    T pi_4 = T(detail::KER_PIO4);

    return -(log_x_half + c10::complex<T>(gamma, T(0))) * ber_sum
           + c10::complex<T>(pi_4, T(0)) * bei_sum + corr_sum;
}

} // namespace torchscience::kernel::special_functions
