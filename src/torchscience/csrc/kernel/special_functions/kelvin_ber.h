#pragma once

#include <cmath>
#include <limits>
#include <c10/util/complex.h>

namespace torchscience::kernel::special_functions {

namespace detail {

// Constants for Kelvin functions
constexpr double KELVIN_SQRT2 = 1.4142135623730950488016887;  // sqrt(2)
constexpr double KELVIN_PI = 3.14159265358979323846264338;
constexpr double KELVIN_PIO8 = 0.39269908169872415480783;  // pi/8

// Threshold for switching between power series and asymptotic expansion
// Power series converges for all x, but convergence is slow for large x
// For x > 20, asymptotic is more efficient
constexpr double KELVIN_BER_LARGE_X = 20.0;

} // namespace detail

// Kelvin function ber(x)
// ber(x) = Re[J_0(x * e^(3*pi*i/4))]
//
// The power series is derived from J_0(z) = sum_{k=0}^inf (-1)^k (z/2)^(2k) / (k!)^2
// With z = x * e^(3*pi*i/4), we get:
//
// J_0(x * e^(3*pi*i/4)) = sum_{k=0}^inf (-1)^k * (x/2)^(2k) * e^(3*k*pi*i/2) / (k!)^2
//
// The factor e^(3*k*pi*i/2) cycles as: 1, -i, -1, i for k = 0, 1, 2, 3, ...
//
// Taking the real part:
// ber(x) = sum_{k=0}^inf Re[(-1)^k * e^(3*k*pi*i/2)] * (x/2)^(2k) / (k!)^2
//
// For k % 4 == 0: (-1)^k * Re[e^0] = 1
// For k % 4 == 1: (-1)^k * Re[e^(3*pi*i/2)] = -1 * 0 = 0
// For k % 4 == 2: (-1)^k * Re[e^(3*pi*i)] = 1 * (-1) = -1
// For k % 4 == 3: (-1)^k * Re[e^(9*pi*i/2)] = -1 * 0 = 0
//
// So: ber(x) = (x/2)^0/(0!)^2 - (x/2)^4/(2!)^2 + (x/2)^8/(4!)^2 - (x/2)^12/(6!)^2 + ...
//            = sum_{n=0}^inf (-1)^n * (x/2)^(4n) / ((2n)!)^2
//
template <typename T>
T kelvin_ber(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // ber is even function: ber(-x) = ber(x)
    x = std::abs(x);

    if (std::isinf(x)) {
        // For large x, ber(x) oscillates with growing amplitude
        return std::numeric_limits<T>::infinity();
    }

    if (x == T(0)) {
        return T(1);
    }

    if (x <= T(detail::KELVIN_BER_LARGE_X)) {
        // Power series for small to medium x
        // ber(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n) / ((2n)!)^2
        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        T sum = T(1);
        T term = T(1);

        for (int n = 1; n < 100; ++n) {
            // Compute ((2n)!)^2 incrementally
            // (2n)! = (2n-2)! * (2n-1) * (2n)
            // So term_n / term_{n-1} = (-1) * x^4 / ((2n-1)^2 * (2n)^2)
            T factor = T(2 * n - 1) * T(2 * n);
            term = -term * x4 / (factor * factor);
            sum += term;

            if (std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
                break;
            }
        }

        return sum;
    } else {
        // Asymptotic expansion for large x
        // ber(x) ~ e^(x/sqrt(2)) / sqrt(2*pi*x) * [f(x) * cos(alpha) + g(x) * sin(alpha)]
        // where alpha = x/sqrt(2) - pi/8
        //
        // The asymptotic expansion coefficients (Abramowitz & Stegun 9.10):
        // f(x) = 1 + sum_{k=1}^inf f_k / x^k
        // g(x) = sum_{k=1}^inf g_k / x^k
        //
        // First several terms of the asymptotic expansion for Kelvin functions:

        T sqrt2 = T(detail::KELVIN_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 - T(detail::KELVIN_PIO8);

        // Compute 1/x powers for asymptotic expansion
        T inv_x = T(1) / x;

        // Asymptotic series from Abramowitz & Stegun 9.10.7-9.10.10
        // These coefficients are for the modulus-phase form
        // M(x) ~ e^(x/sqrt(2)) / sqrt(2*pi*x), theta(x) ~ x/sqrt(2) - pi/8
        //
        // More precisely:
        // ber(x) = M * cos(theta - phi)
        // bei(x) = M * sin(theta - phi)
        //
        // where phi has an asymptotic expansion. For accuracy we need
        // the asymptotic expansion of ber directly:
        //
        // ber(x) ~ (e^(x/sqrt(2)) / sqrt(2*pi*x)) * {
        //   [1 - 1/(8*sqrt(2)*x) - 1/(256*x^2) - ...] * cos(x/sqrt(2) - pi/8)
        //   + [1/(8*sqrt(2)*x) - 1/(256*x^2) + ...] * sin(x/sqrt(2) - pi/8)
        // }

        // Coefficients for f and g series (from numerical fitting/standard tables)
        // f = 1 + f1/x + f2/x^2 + f3/x^3 + ...
        // g = g1/x + g2/x^2 + g3/x^3 + ...

        T f1 = T(-1) / (T(8) * sqrt2);       // -1/(8*sqrt(2))
        T g1 = T(1) / (T(8) * sqrt2);        // 1/(8*sqrt(2))
        T f2 = T(-1) / T(256);               // -1/256
        T g2 = T(-1) / T(256);               // -1/256
        T f3 = T(25) / (T(3072) * sqrt2);    // 25/(3072*sqrt(2))
        T g3 = T(-13) / (T(3072) * sqrt2);   // -13/(3072*sqrt(2))
        T f4 = T(13) / T(16384);             // 13/16384
        T g4 = T(-13) / T(16384);            // -13/16384

        T inv_x2 = inv_x * inv_x;
        T inv_x3 = inv_x2 * inv_x;
        T inv_x4 = inv_x2 * inv_x2;

        T f = T(1) + f1 * inv_x + f2 * inv_x2 + f3 * inv_x3 + f4 * inv_x4;
        T g = g1 * inv_x + g2 * inv_x2 + g3 * inv_x3 + g4 * inv_x4;

        T exp_factor = std::exp(x_over_sqrt2);
        T sqrt_factor = T(1) / std::sqrt(T(2) * T(detail::KELVIN_PI) * x);

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        return exp_factor * sqrt_factor * (f * cos_alpha + g * sin_alpha);
    }
}

// Complex version
template <typename T>
c10::complex<T> kelvin_ber(c10::complex<T> x) {
    // For complex x, we use the power series which converges for all finite x
    // ber(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n) / ((2n)!)^2

    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    c10::complex<T> sum(T(1), T(0));
    c10::complex<T> term(T(1), T(0));

    for (int n = 1; n < 60; ++n) {
        T factor = T(2 * n - 1) * T(2 * n);
        term = -term * x4 / c10::complex<T>(factor * factor, T(0));
        sum = sum + term;

        if (std::abs(term) < std::numeric_limits<T>::epsilon() * std::abs(sum)) {
            break;
        }
    }

    return sum;
}

} // namespace torchscience::kernel::special_functions
