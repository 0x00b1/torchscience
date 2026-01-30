#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "kelvin_bei.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Kelvin function bei'(x) - derivative of bei(x)
//
// From bei(x) = sum_{n=0}^inf (-1)^n * (x/2)^(4n+2) / ((2n+1)!)^2
// bei'(x) = sum_{n=0}^inf (-1)^n * (4n+2) * (x/2)^(4n+1) / (2 * ((2n+1)!)^2)
//         = sum_{n=0}^inf (-1)^n * (2n+1) * (x/2)^(4n+1) / ((2n+1)!)^2
//         = (x/2) * sum_{n=0}^inf (-1)^n * (2n+1) / ((2n+1)!)^2 * (x/2)^(4n)
//
template <typename T>
T kelvin_bei_derivative(T x) {
    // Handle special values
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    if (x == T(0)) {
        return T(0);  // bei'(0) = 0
    }

    bool negative = x < T(0);
    x = std::abs(x);

    if (std::isinf(x)) {
        return std::numeric_limits<T>::infinity();
    }

    T result;

    if (x <= T(20.0)) {
        // Power series for the derivative
        // bei'(x) = sum_{n=0}^inf (-1)^n * (2n+1) / ((2n+1)!)^2 * (x/2)^(4n+1)
        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        // First term (n=0): 1 / (1!)^2 * (x/2) = x/2
        T sum = T(0);
        T factorial_2np1_sq = T(1);  // ((2n+1)!)^2

        for (int n = 0; n < 100; ++n) {
            // Compute (2n+1)!^2 incrementally
            if (n > 0) {
                // (2n+1)! = (2(n-1)+1)! * (2n) * (2n+1)
                T f1 = T(2 * n);
                T f2 = T(2 * n + 1);
                factorial_2np1_sq *= (f1 * f2) * (f1 * f2);
            }

            // term = (-1)^n * (2n+1) / ((2n+1)!)^2 * (x/2)^(4n+1)
            T sign = (n % 2 == 0) ? T(1) : T(-1);
            T coeff = sign * T(2 * n + 1) / factorial_2np1_sq;
            T x_power = std::pow(x_half, T(4 * n + 1));
            T term_n = coeff * x_power;

            sum += term_n;

            if (std::abs(term_n) < std::numeric_limits<T>::epsilon() * std::abs(sum) && n > 3) {
                break;
            }
        }

        result = sum;
    } else {
        // Asymptotic expansion for derivative
        // d/dx bei(x) from the asymptotic form
        T sqrt2 = T(KELVIN_BEI_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 - T(KELVIN_BEI_PIO8);

        T exp_factor = std::exp(x_over_sqrt2);
        T sqrt_factor = T(1) / std::sqrt(T(2) * T(KELVIN_BEI_PI) * x);

        // For the derivative, we differentiate:
        // bei(x) = exp(x/sqrt2) / sqrt(2*pi*x) * [f*sin(alpha) - g*cos(alpha)]
        //
        // Leading order approximation:
        // d/dx bei(x) ~ (exp/sqrt) / sqrt2 * [cos(alpha) + sin(alpha)]

        T inv_sqrt2 = T(1) / sqrt2;
        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        // Full expression with leading correction terms
        T inv_x = T(1) / x;

        T f1 = T(-1) / (T(8) * sqrt2);
        T g1 = T(1) / (T(8) * sqrt2);

        T f = T(1) + f1 * inv_x;
        T g = g1 * inv_x;

        // Main derivative contributions
        T term1 = inv_sqrt2 * (f * sin_alpha - g * cos_alpha);  // from exp derivative
        T term2 = inv_sqrt2 * (f * cos_alpha + g * sin_alpha);  // from alpha derivative
        T term3 = -T(0.5) * inv_x * (f * sin_alpha - g * cos_alpha);  // from 1/sqrt(x) derivative

        result = exp_factor * sqrt_factor * (term1 + term2 + term3);
    }

    // bei(-x) = bei(x), so bei'(-x) = -bei'(x)
    return negative ? -result : result;
}

} // namespace detail

// Real backward: d/dx bei(x)
template <typename T>
T kelvin_bei_backward(T grad_output, T x) {
    return grad_output * detail::kelvin_bei_derivative(x);
}

// Complex backward
template <typename T>
c10::complex<T> kelvin_bei_backward(c10::complex<T> grad_output, c10::complex<T> x) {
    // Compute derivative using power series
    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    c10::complex<T> sum(T(0), T(0));
    T factorial_2np1_sq = T(1);

    for (int n = 0; n < 60; ++n) {
        if (n > 0) {
            T f1 = T(2 * n);
            T f2 = T(2 * n + 1);
            factorial_2np1_sq *= (f1 * f2) * (f1 * f2);
        }

        T sign = (n % 2 == 0) ? T(1) : T(-1);
        T coeff = sign * T(2 * n + 1) / factorial_2np1_sq;

        // (x/2)^(4n+1)
        c10::complex<T> x_power = std::pow(x_half, T(4 * n + 1));
        c10::complex<T> term_n = c10::complex<T>(coeff, T(0)) * x_power;

        sum = sum + term_n;

        if (std::abs(term_n) < std::numeric_limits<T>::epsilon() * std::abs(sum) && n > 3) {
            break;
        }
    }

    return grad_output * std::conj(sum);
}

} // namespace torchscience::kernel::special_functions
