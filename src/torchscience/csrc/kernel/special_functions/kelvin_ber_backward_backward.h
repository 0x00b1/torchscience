#pragma once

#include <cmath>
#include <tuple>
#include <c10/util/complex.h>
#include "kelvin_ber.h"
#include "kelvin_ber_backward.h"

namespace torchscience::kernel::special_functions {

namespace detail {

// Second derivative of Kelvin function ber(x)
//
// From ber'(x) = sum_{n=1}^inf (-1)^n * 2n / ((2n)!)^2 * (x/2)^(4n-1)
// ber''(x) = sum_{n=1}^inf (-1)^n * 2n * (4n-1) / ((2n)!)^2 * (x/2)^(4n-2) / 2
//          = sum_{n=1}^inf (-1)^n * n * (4n-1) / ((2n)!)^2 * (x/2)^(4n-2)

template <typename T>
T kelvin_ber_second_derivative(T x) {
    if (std::isnan(x)) {
        return std::numeric_limits<T>::quiet_NaN();
    }

    // ber is even, so ber'' is also even
    x = std::abs(x);

    if (std::isinf(x)) {
        return std::numeric_limits<T>::infinity();
    }

    if (x == T(0)) {
        // ber''(0) = 0 from the series (first nonzero term is x^2)
        return T(0);
    }

    if (x <= T(20.0)) {
        // Power series for second derivative
        T x_half = x / T(2);
        T x2 = x_half * x_half;
        T x4 = x2 * x2;

        T sum = T(0);
        T factorial_2n_sq = T(1);

        for (int n = 1; n < 100; ++n) {
            T f1 = T(2 * n - 1);
            T f2 = T(2 * n);
            factorial_2n_sq *= (f1 * f2) * (f1 * f2);

            T sign = (n % 2 == 1) ? T(-1) : T(1);
            // Coefficient: n * (4n-1) / ((2n)!)^2
            T coeff = sign * T(n) * T(4 * n - 1) / factorial_2n_sq;

            // (x/2)^(4n-2)
            T x_power = std::pow(x_half, T(4 * n - 2));
            T term_n = coeff * x_power;

            sum += term_n;

            if (std::abs(term_n) < std::numeric_limits<T>::epsilon() * std::abs(sum) && n > 3) {
                break;
            }
        }

        return sum;
    } else {
        // Asymptotic expansion for second derivative
        T sqrt2 = T(KELVIN_SQRT2);
        T x_over_sqrt2 = x / sqrt2;
        T alpha = x_over_sqrt2 - T(KELVIN_PIO8);

        T exp_factor = std::exp(x_over_sqrt2);
        T sqrt_factor = T(1) / std::sqrt(T(2) * T(KELVIN_PI) * x);

        T cos_alpha = std::cos(alpha);
        T sin_alpha = std::sin(alpha);

        // Leading order: ~ exp/sqrt * (1/2) * [-cos(alpha) - sin(alpha)]
        T leading = T(0.5) * (-cos_alpha - sin_alpha);

        return exp_factor * sqrt_factor * leading;
    }
}

} // namespace detail

// Real backward_backward
// Returns gradients for (grad_output, x)
template <typename T>
std::tuple<T, T> kelvin_ber_backward_backward(T gg_x, T grad_output, T x) {
    T first_deriv = detail::kelvin_ber_derivative(x);
    T second_deriv = detail::kelvin_ber_second_derivative(x);

    // d(backward)/d(grad_output) = ber'(x)
    T grad_grad_output = gg_x * first_deriv;

    // d(backward)/dx = grad_output * ber''(x)
    T grad_x = gg_x * grad_output * second_deriv;

    return {grad_grad_output, grad_x};
}

// Complex backward_backward
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> kelvin_ber_backward_backward(
    c10::complex<T> gg_x, c10::complex<T> grad_output, c10::complex<T> x) {

    // Compute first derivative using power series
    c10::complex<T> x_half = x / c10::complex<T>(T(2), T(0));
    c10::complex<T> x2 = x_half * x_half;
    c10::complex<T> x4 = x2 * x2;

    c10::complex<T> sum1(T(0), T(0));
    T factorial_2n_sq1 = T(1);

    for (int n = 1; n < 60; ++n) {
        T f1 = T(2 * n - 1);
        T f2 = T(2 * n);
        factorial_2n_sq1 *= (f1 * f2) * (f1 * f2);

        T sign = (n % 2 == 1) ? T(-1) : T(1);
        T coeff = sign * T(2 * n) / factorial_2n_sq1;

        c10::complex<T> x_power = std::pow(x_half, T(4 * n - 1));
        c10::complex<T> term_n = c10::complex<T>(coeff, T(0)) * x_power;

        sum1 = sum1 + term_n;

        if (std::abs(term_n) < std::numeric_limits<T>::epsilon() * std::abs(sum1) && n > 3) {
            break;
        }
    }
    c10::complex<T> first_deriv = sum1;

    // Compute second derivative using power series
    c10::complex<T> sum2(T(0), T(0));
    T factorial_2n_sq2 = T(1);

    for (int n = 1; n < 60; ++n) {
        T f1 = T(2 * n - 1);
        T f2 = T(2 * n);
        factorial_2n_sq2 *= (f1 * f2) * (f1 * f2);

        T sign = (n % 2 == 1) ? T(-1) : T(1);
        T coeff = sign * T(n) * T(4 * n - 1) / factorial_2n_sq2;

        c10::complex<T> x_power = std::pow(x_half, T(4 * n - 2));
        c10::complex<T> term_n = c10::complex<T>(coeff, T(0)) * x_power;

        sum2 = sum2 + term_n;

        if (std::abs(term_n) < std::numeric_limits<T>::epsilon() * std::abs(sum2) && n > 3) {
            break;
        }
    }
    c10::complex<T> second_deriv = sum2;

    c10::complex<T> grad_grad_output = gg_x * std::conj(first_deriv);
    c10::complex<T> grad_x = gg_x * grad_output * std::conj(second_deriv);

    return {grad_grad_output, grad_x};
}

} // namespace torchscience::kernel::special_functions
