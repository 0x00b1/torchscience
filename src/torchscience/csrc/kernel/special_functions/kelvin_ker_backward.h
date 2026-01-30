#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "kelvin_ker.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Kelvin function ker'(x) - derivative of ker(x)
//
// The derivative can be expressed in terms of ker_1 and kei_1, but for numerical
// stability we use direct differentiation of the series and asymptotic expansions.
//
// For small x, from ker(x) = -(ln(x/2) + gamma) * ber(x) + (pi/4) * bei(x) + corr_sum:
// ker'(x) = -(1/x) * ber(x) - (ln(x/2) + gamma) * ber'(x) + (pi/4) * bei'(x) + corr_sum'
//
// For large x, we differentiate the asymptotic expansion.
//
template <typename T>
T kelvin_ker_derivative(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x == T(0)) {
        // ker'(0) = -infinity (from the -1/x * ber(x) term, since ber(0) = 1)
        return -std::numeric_limits<T>::infinity();
    }

    bool negative = x < T(0);
    x = std::abs(x);

    if (std::isinf(x)) {
        return T(0);  // Exponential decay dominates
    }

    T result;

    if (x <= T(KER_LARGE_X)) {
        // Power series for the derivative
        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        // Compute ber(x), ber'(x), bei(x), bei'(x), and correction derivatives

        // ber(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n) / ((2n)!)^2
        T ber_sum = T(1);
        T ber_term = T(1);

        // ber'(x) = sum_{n=1}^inf (-1)^n * 4n * (x/2)^(4n-1) / (2 * ((2n)!)^2)
        //         = sum_{n=1}^inf (-1)^n * 2n * (x/2)^(4n-1) / ((2n)!)^2
        T ber_deriv_sum = T(0);
        T factorial_2n_sq = T(1);

        // bei(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n+2) / ((2n+1)!)^2
        T bei_sum = x2;
        T bei_term = x2;

        // bei'(x) = sum_{n=0}^inf (-1)^n * (4n+2) * (x/2)^(4n+1) / (2 * ((2n+1)!)^2)
        T bei_deriv_sum = x_half;  // First term: (x/2)^1

        // Correction series: sum_{k=0}^inf [(-1)^k / ((2k)!)^2] * H_{2k} * (x/2)^(4k)
        // Derivative: sum_{k=1}^inf [(-1)^k / ((2k)!)^2] * H_{2k} * 4k * (x/2)^(4k-1) / 2
        T corr_deriv_sum = T(0);
        T harmonic = T(0);

        for (int n = 1; n < 100; ++n) {
            T factor = T(2 * n - 1) * T(2 * n);
            factorial_2n_sq *= factor * factor;
            harmonic += T(1) / T(2 * n - 1) + T(1) / T(2 * n);

            // ber term
            ber_term = -ber_term * x4 / (factor * factor);
            ber_sum += ber_term;

            // ber' term: (-1)^n * 2n / ((2n)!)^2 * (x/2)^(4n-1)
            T sign = (n % 2 == 1) ? T(-1) : T(1);
            T x_power_4n_1 = std::pow(x_half, T(4 * n - 1));
            T ber_deriv_term = sign * T(2 * n) / factorial_2n_sq * x_power_4n_1;
            ber_deriv_sum += ber_deriv_term;

            // bei term
            T bei_factor = T(2 * n) * T(2 * n + 1);
            bei_term = -bei_term * x4 / (bei_factor * bei_factor);
            bei_sum += bei_term;

            // bei' term: (-1)^n * (4n+2) / (2 * ((2n+1)!)^2) * (x/2)^(4n+1)
            T factorial_2n1_sq = factorial_2n_sq * T(2 * n + 1) * T(2 * n + 1);
            T x_power_4n_1_bei = std::pow(x_half, T(4 * n + 1));
            T bei_deriv_term = sign * T(4 * n + 2) / (T(2) * factorial_2n1_sq) * x_power_4n_1_bei;
            bei_deriv_sum += bei_deriv_term;

            // Correction derivative term: (-1)^n * H_{2n} * 2n / ((2n)!)^2 * (x/2)^(4n-1)
            T corr_deriv_term = sign * harmonic * T(2 * n) / factorial_2n_sq * x_power_4n_1;
            corr_deriv_sum += corr_deriv_term;

            if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
                std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
                n > 5) {
                break;
            }
        }

        // ker'(x) = -(1/x) * ber(x) - (ln(x/2) + gamma) * ber'(x) + (pi/4) * bei'(x) + corr'
        T log_x_half = std::log(x_half);
        T gamma = T(KER_EULER_GAMMA);
        T pi_4 = T(KER_PIO4);

        result = -(T(1) / x) * ber_sum - (log_x_half + gamma) * ber_deriv_sum
                 + pi_4 * bei_deriv_sum + corr_deriv_sum;
    } else {
        // Asymptotic expansion for derivative
        // d/dx[sqrt(pi/(2x)) * exp(-x/sqrt(2)) * (f*cos(alpha) + g*sin(alpha))]
        //
        // Let M = sqrt(pi/(2x)) * exp(-x/sqrt(2))
        // ker = M * [f*cos(alpha) + g*sin(alpha)]
        //
        // ker' = M' * [...] + M * [f'*cos(alpha) - f*sin(alpha)*alpha' + g'*sin(alpha) + g*cos(alpha)*alpha']
        // M' = M * (-1/(2x) - 1/sqrt(2))
        // alpha' = 1/sqrt(2)

        T sqrt2 = T(KER_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 + T(KER_PIO8);
        T inv_sqrt2 = T(1) / sqrt2;

        T inv_x = T(1) / x;
        T inv_x2 = inv_x * inv_x;
        T inv_x3 = inv_x2 * inv_x;
        T inv_x4 = inv_x2 * inv_x2;

        // Asymptotic coefficients
        T f1 = T(-1) / (T(8) * sqrt2);
        T g1 = T(-1) / (T(8) * sqrt2);
        T f2 = T(1) / T(256);
        T g2 = T(-1) / T(256);
        T f3 = T(25) / (T(3072) * sqrt2);
        T g3 = T(13) / (T(3072) * sqrt2);
        T f4 = T(-13) / T(16384);
        T g4 = T(-13) / T(16384);

        T f = T(1) + f1 * inv_x + f2 * inv_x2 + f3 * inv_x3 + f4 * inv_x4;
        T g = g1 * inv_x + g2 * inv_x2 + g3 * inv_x3 + g4 * inv_x4;

        T exp_factor = std::exp(-x_over_sqrt2);
        T sqrt_factor = std::sqrt(T(KER_PI) / (T(2) * x));

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        // Main term coefficient and its derivative
        T M = exp_factor * sqrt_factor;
        T dM_dx = M * (-T(0.5) * inv_x - inv_sqrt2);

        // Combined expression
        T inner = f * cos_alpha + g * sin_alpha;
        T d_inner_dalpha = -f * sin_alpha + g * cos_alpha;

        // Neglecting f' and g' contributions (they are O(1/x^2))
        result = dM_dx * inner + M * d_inner_dalpha * inv_sqrt2;
    }

    // ker is even, so ker'(-x) = -ker'(x)
    return negative ? -result : result;
}

} // namespace detail

// Real backward: d/dx ker(x)
template <typename T>
T kelvin_ker_backward(T grad_output, T x) {
    return grad_output * detail::kelvin_ker_derivative(x);
}

// Complex backward
template <typename T>
c10::complex<T> kelvin_ker_backward(c10::complex<T> grad_output, c10::complex<T> x) {
    // Compute derivative using power series
    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    // Compute ber, ber', bei, bei', and correction derivatives
    c10::complex<T> ber_sum(T(1), T(0));
    c10::complex<T> ber_term(T(1), T(0));
    c10::complex<T> ber_deriv_sum(T(0), T(0));

    c10::complex<T> bei_sum = x2;
    c10::complex<T> bei_term = x2;
    c10::complex<T> bei_deriv_sum = x_half;

    c10::complex<T> corr_deriv_sum(T(0), T(0));

    T harmonic = T(0);
    T factorial_2n_sq = T(1);

    for (int n = 1; n < 60; ++n) {
        T factor = T(2 * n - 1) * T(2 * n);
        factorial_2n_sq *= factor * factor;
        harmonic += T(1) / T(2 * n - 1) + T(1) / T(2 * n);

        // ber term
        ber_term = -ber_term * x4 / c10::complex<T>(factor * factor, T(0));
        ber_sum = ber_sum + ber_term;

        // ber' term
        T sign = (n % 2 == 1) ? T(-1) : T(1);
        c10::complex<T> x_power_4n_1 = std::pow(x_half, T(4 * n - 1));
        c10::complex<T> ber_deriv_term = c10::complex<T>(sign * T(2 * n) / factorial_2n_sq, T(0)) * x_power_4n_1;
        ber_deriv_sum = ber_deriv_sum + ber_deriv_term;

        // bei term
        T bei_factor = T(2 * n) * T(2 * n + 1);
        bei_term = -bei_term * x4 / c10::complex<T>(bei_factor * bei_factor, T(0));
        bei_sum = bei_sum + bei_term;

        // bei' term
        T factorial_2n1_sq = factorial_2n_sq * T(2 * n + 1) * T(2 * n + 1);
        c10::complex<T> x_power_4n_1_bei = std::pow(x_half, T(4 * n + 1));
        c10::complex<T> bei_deriv_term = c10::complex<T>(sign * T(4 * n + 2) / (T(2) * factorial_2n1_sq), T(0)) * x_power_4n_1_bei;
        bei_deriv_sum = bei_deriv_sum + bei_deriv_term;

        // Correction derivative term
        c10::complex<T> corr_deriv_term = c10::complex<T>(sign * harmonic * T(2 * n) / factorial_2n_sq, T(0)) * x_power_4n_1;
        corr_deriv_sum = corr_deriv_sum + corr_deriv_term;

        if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
            std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
            n > 5) {
            break;
        }
    }

    c10::complex<T> log_x_half = std::log(x_half);
    T gamma = T(detail::KER_EULER_GAMMA);
    T pi_4 = T(detail::KER_PIO4);

    c10::complex<T> derivative = -(c10::complex<T>(T(1), T(0)) / x) * ber_sum
                                  - (log_x_half + c10::complex<T>(gamma, T(0))) * ber_deriv_sum
                                  + c10::complex<T>(pi_4, T(0)) * bei_deriv_sum
                                  + corr_deriv_sum;

    return grad_output * std::conj(derivative);
}

} // namespace torchscience::kernel::special_functions
