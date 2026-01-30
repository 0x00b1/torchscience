#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "kelvin_kei.h"
#include "kelvin_kei_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative of Kelvin function kei(x)
//
// From kei'(x) = -(1/x)*bei(x) - (ln(x/2)+gamma)*bei'(x) - (pi/4)*ber'(x) + corr'
// kei''(x) = (1/x^2)*bei(x) - (1/x)*bei'(x) - (1/x)*bei'(x) - (ln(x/2)+gamma)*bei''(x)
//          - (pi/4)*ber''(x) + corr''
//        = (1/x^2)*bei(x) - (2/x)*bei'(x) - (ln(x/2)+gamma)*bei''(x) - (pi/4)*ber''(x) + corr''

template <typename T>
T kelvin_kei_second_derivative(T x) {
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // kei is even, so kei'' is also even
    x = std::abs(x);

    if (std::isinf(x)) {
        return T(0);
    }

    if (x == T(0)) {
        // kei''(0): Need to evaluate limit carefully
        // bei(x) ~ x^2/4, bei'(x) ~ x/2
        // (1/x^2)*bei(x) ~ 1/4, (2/x)*bei'(x) ~ 1
        // So kei''(0) is finite
        // Numerical limit gives approximately -0.25 (from the pi/4 * ber''(0) = pi/4 * 0 = 0
        // and the bei terms dominate)
        return T(-0.25);
    }

    if (x <= T(KEI_LARGE_X)) {
        // Power series for second derivative
        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        // Compute ber, ber', ber'', bei, bei', bei'', and correction second derivatives
        T ber_sum = T(1);
        T ber_term = T(1);
        T ber_deriv_sum = T(0);
        T ber_second_deriv_sum = T(0);

        T bei_sum = x2;
        T bei_term = x2;
        T bei_deriv_sum = x_half;
        T bei_second_deriv_sum = T(0.5);  // d/dx(x/2) = 1/2

        T corr_second_deriv_sum = T(0.5);  // First term (k=0): H_1 * 1 * (x/2)^0 / 2 = 0.5

        T harmonic = T(1);
        T factorial_2n_sq = T(1);
        T factorial_odd_sq = T(1);

        for (int n = 1; n < 100; ++n) {
            T factor = T(2 * n - 1) * T(2 * n);
            factorial_2n_sq *= factor * factor;

            T sign = (n % 2 == 1) ? T(-1) : T(1);

            // ber term
            ber_term = -ber_term * x4 / (factor * factor);
            ber_sum += ber_term;

            // ber' term: (-1)^n * 2n / ((2n)!)^2 * (x/2)^(4n-1)
            T x_power_4n_1 = std::pow(x_half, T(4 * n - 1));
            T ber_deriv_term = sign * T(2 * n) / factorial_2n_sq * x_power_4n_1;
            ber_deriv_sum += ber_deriv_term;

            // ber'' term: (-1)^n * 2n * (4n-1) / ((2n)!)^2 * (x/2)^(4n-2) / 2
            //           = (-1)^n * n * (4n-1) / ((2n)!)^2 * (x/2)^(4n-2)
            T x_power_4n_2 = std::pow(x_half, T(4 * n - 2));
            T ber_second_deriv_term = sign * T(n) * T(4 * n - 1) / factorial_2n_sq * x_power_4n_2;
            ber_second_deriv_sum += ber_second_deriv_term;

            // bei term
            T bei_factor = T(2 * n) * T(2 * n + 1);
            bei_term = -bei_term * x4 / (bei_factor * bei_factor);
            bei_sum += bei_term;

            // bei' term
            T factorial_2n1_sq = factorial_2n_sq * T(2 * n + 1) * T(2 * n + 1);
            T x_power_4n_1_bei = std::pow(x_half, T(4 * n + 1));
            T bei_deriv_term = sign * T(4 * n + 2) / (T(2) * factorial_2n1_sq) * x_power_4n_1_bei;
            bei_deriv_sum += bei_deriv_term;

            // bei'' term: (-1)^n * (4n+2) * (4n+1) / (2 * 2 * ((2n+1)!)^2) * (x/2)^(4n)
            //           = (-1)^n * (4n+2) * (4n+1) / (4 * ((2n+1)!)^2) * (x/2)^(4n)
            T x_power_4n = std::pow(x_half, T(4 * n));
            T bei_second_deriv_term = sign * T(4 * n + 2) * T(4 * n + 1) / (T(4) * factorial_2n1_sq) * x_power_4n;
            bei_second_deriv_sum += bei_second_deriv_term;

            // Correction second derivative term
            // From corr'(x) = sum ... * (2n+1) * (x/2)^(4n+1)
            // corr''(x) = sum ... * (2n+1) * (4n+1) * (x/2)^(4n) / 2
            factorial_odd_sq *= T(2 * n) * T(2 * n) * T(2 * n + 1) * T(2 * n + 1);
            harmonic += T(1) / T(2 * n) + T(1) / T(2 * n + 1);
            T corr_second_deriv_term = sign * harmonic * T(2 * n + 1) * T(4 * n + 1) / (T(2) * factorial_odd_sq) * x_power_4n;
            corr_second_deriv_sum += corr_second_deriv_term;

            if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
                std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
                n > 5) {
                break;
            }
        }

        T log_x_half = std::log(x_half);
        T gamma = T(KEI_EULER_GAMMA);
        T pi_4 = T(KEI_PIO4);
        T inv_x = T(1) / x;
        T inv_x2 = inv_x * inv_x;

        // kei''(x) = (1/x^2)*bei(x) - (2/x)*bei'(x) - (ln(x/2)+gamma)*bei''(x)
        //          - (pi/4)*ber''(x) + corr''
        return inv_x2 * bei_sum - T(2) * inv_x * bei_deriv_sum
               - (log_x_half + gamma) * bei_second_deriv_sum
               - pi_4 * ber_second_deriv_sum + corr_second_deriv_sum;
    } else {
        // Asymptotic expansion for second derivative
        T sqrt2 = T(KEI_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 + T(KEI_PIO8);
        T inv_sqrt2 = T(1) / sqrt2;

        T exp_factor = std::exp(-x_over_sqrt2);
        T sqrt_factor = std::sqrt(T(KEI_PI) / (T(2) * x));

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        // Leading order approximation for kei''
        T inv_x = T(1) / x;

        // Simplified leading order approximation
        T leading = (T(0.5) + inv_sqrt2 * inv_sqrt2) * sin_alpha
                  - (inv_sqrt2 + T(0.5) * inv_sqrt2) * cos_alpha;

        return -exp_factor * sqrt_factor * leading;
    }
}

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, x)
template <typename T>
std::tuple<T, T> kelvin_kei_backward_backward(T gg_x, T grad_output, T x) {
    T first_deriv = detail::kelvin_kei_derivative(x);
    T second_deriv = detail::kelvin_kei_second_derivative(x);

    // d(backward)/d(grad_output) = kei'(x)
    T grad_grad_output = gg_x * first_deriv;

    // d(backward)/dx = grad_output * kei''(x)
    T grad_x = gg_x * grad_output * second_deriv;

    return {grad_grad_output, grad_x};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> kelvin_kei_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> grad_output, c10::complex<T> x) {

    // Compute first and second derivatives using power series
    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    c10::complex<T> ber_sum(T(1), T(0));
    c10::complex<T> ber_term(T(1), T(0));
    c10::complex<T> ber_deriv_sum(T(0), T(0));
    c10::complex<T> ber_second_deriv_sum(T(0), T(0));

    c10::complex<T> bei_sum = x2;
    c10::complex<T> bei_term = x2;
    c10::complex<T> bei_deriv_sum = x_half;
    c10::complex<T> bei_second_deriv_sum(T(0.5), T(0));

    c10::complex<T> corr_deriv_sum = x_half;
    c10::complex<T> corr_second_deriv_sum(T(0.5), T(0));

    T harmonic = T(1);
    T factorial_2n_sq = T(1);
    T factorial_odd_sq = T(1);

    for (int n = 1; n < 60; ++n) {
        T factor = T(2 * n - 1) * T(2 * n);
        factorial_2n_sq *= factor * factor;

        T sign = (n % 2 == 1) ? T(-1) : T(1);

        // ber terms
        ber_term = -ber_term * x4 / c10::complex<T>(factor * factor, T(0));
        ber_sum = ber_sum + ber_term;

        c10::complex<T> x_power_4n_1 = std::pow(x_half, T(4 * n - 1));
        c10::complex<T> ber_deriv_term = c10::complex<T>(sign * T(2 * n) / factorial_2n_sq, T(0)) * x_power_4n_1;
        ber_deriv_sum = ber_deriv_sum + ber_deriv_term;

        c10::complex<T> x_power_4n_2 = std::pow(x_half, T(4 * n - 2));
        c10::complex<T> ber_second_deriv_term = c10::complex<T>(sign * T(n) * T(4 * n - 1) / factorial_2n_sq, T(0)) * x_power_4n_2;
        ber_second_deriv_sum = ber_second_deriv_sum + ber_second_deriv_term;

        // bei terms
        T bei_factor = T(2 * n) * T(2 * n + 1);
        bei_term = -bei_term * x4 / c10::complex<T>(bei_factor * bei_factor, T(0));
        bei_sum = bei_sum + bei_term;

        T factorial_2n1_sq = factorial_2n_sq * T(2 * n + 1) * T(2 * n + 1);
        c10::complex<T> x_power_4n_1_bei = std::pow(x_half, T(4 * n + 1));
        c10::complex<T> bei_deriv_term = c10::complex<T>(sign * T(4 * n + 2) / (T(2) * factorial_2n1_sq), T(0)) * x_power_4n_1_bei;
        bei_deriv_sum = bei_deriv_sum + bei_deriv_term;

        c10::complex<T> x_power_4n = std::pow(x_half, T(4 * n));
        c10::complex<T> bei_second_deriv_term = c10::complex<T>(sign * T(4 * n + 2) * T(4 * n + 1) / (T(4) * factorial_2n1_sq), T(0)) * x_power_4n;
        bei_second_deriv_sum = bei_second_deriv_sum + bei_second_deriv_term;

        // Correction terms
        factorial_odd_sq *= T(2 * n) * T(2 * n) * T(2 * n + 1) * T(2 * n + 1);
        harmonic += T(1) / T(2 * n) + T(1) / T(2 * n + 1);
        c10::complex<T> corr_deriv_term = c10::complex<T>(sign * harmonic * T(2 * n + 1) / factorial_odd_sq, T(0)) * x_power_4n_1_bei;
        corr_deriv_sum = corr_deriv_sum + corr_deriv_term;

        c10::complex<T> corr_second_deriv_term = c10::complex<T>(sign * harmonic * T(2 * n + 1) * T(4 * n + 1) / (T(2) * factorial_odd_sq), T(0)) * x_power_4n;
        corr_second_deriv_sum = corr_second_deriv_sum + corr_second_deriv_term;

        if (std::abs(ber_term) < std::numeric_limits<T>::epsilon() * std::abs(ber_sum) &&
            std::abs(bei_term) < std::numeric_limits<T>::epsilon() * std::abs(bei_sum) &&
            n > 5) {
            break;
        }
    }

    c10::complex<T> log_x_half = std::log(x_half);
    T gamma = T(detail::KEI_EULER_GAMMA);
    T pi_4 = T(detail::KEI_PIO4);
    c10::complex<T> inv_x = c10::complex<T>(T(1), T(0)) / x;
    c10::complex<T> inv_x2 = inv_x * inv_x;

    c10::complex<T> first_deriv = -inv_x * bei_sum
                                   - (log_x_half + c10::complex<T>(gamma, T(0))) * bei_deriv_sum
                                   - c10::complex<T>(pi_4, T(0)) * ber_deriv_sum
                                   + corr_deriv_sum;

    c10::complex<T> second_deriv = inv_x2 * bei_sum
                                    - c10::complex<T>(T(2), T(0)) * inv_x * bei_deriv_sum
                                    - (log_x_half + c10::complex<T>(gamma, T(0))) * bei_second_deriv_sum
                                    - c10::complex<T>(pi_4, T(0)) * ber_second_deriv_sum
                                    + corr_second_deriv_sum;

    c10::complex<T> grad_grad_output = gg_x * std::conj(first_deriv);
    c10::complex<T> grad_x = gg_x * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_x};
}

} // namespace torchscience::kernel::special_functions
