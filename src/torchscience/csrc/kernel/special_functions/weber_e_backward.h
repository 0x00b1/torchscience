#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "weber_e.h"

namespace torchscience::kernel::special_functions {

// Helper: compute d/dn E_nu(z) = (1/pi) * integral_0^pi theta * cos(nu*theta - z*sin(theta)) d(theta)
// using Gauss-Legendre quadrature
template <typename T>
T weber_e_derivative_n(T n, T z) {
    const T pi = T(3.14159265358979323846);

    // 20-point Gauss-Legendre quadrature on [0, pi]
    const T nodes[10] = {
        T(0.0765265211334973),
        T(0.2277858511416451),
        T(0.3737060887154195),
        T(0.5108670019508271),
        T(0.6360536807265150),
        T(0.7463319064601508),
        T(0.8391169718222188),
        T(0.9122344282513259),
        T(0.9639719272779138),
        T(0.9931285991850949)
    };

    const T weights[10] = {
        T(0.1527533871307258),
        T(0.1491729864726037),
        T(0.1420961093183820),
        T(0.1316886384491766),
        T(0.1181945319615184),
        T(0.1019301198172404),
        T(0.0832767415767048),
        T(0.0626720483341091),
        T(0.0406014298003869),
        T(0.0176140071391521)
    };

    T result = T(0);
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        // Positive node
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        T arg_pos = n * theta_pos - z * std::sin(theta_pos);
        // d/dn integrand has extra factor of theta
        T integrand_pos = theta_pos * std::cos(arg_pos);
        result += weights[j] * integrand_pos;

        // Negative node
        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        T arg_neg = n * theta_neg - z * std::sin(theta_neg);
        T integrand_neg = theta_neg * std::cos(arg_neg);
        result += weights[j] * integrand_neg;
    }

    return result * half_pi / pi;
}

// Complex version of derivative w.r.t. n
template <typename T>
c10::complex<T> weber_e_derivative_n(c10::complex<T> n, c10::complex<T> z) {
    const T pi = T(3.14159265358979323846);

    const T nodes[10] = {
        T(0.0765265211334973),
        T(0.2277858511416451),
        T(0.3737060887154195),
        T(0.5108670019508271),
        T(0.6360536807265150),
        T(0.7463319064601508),
        T(0.8391169718222188),
        T(0.9122344282513259),
        T(0.9639719272779138),
        T(0.9931285991850949)
    };

    const T weights[10] = {
        T(0.1527533871307258),
        T(0.1491729864726037),
        T(0.1420961093183820),
        T(0.1316886384491766),
        T(0.1181945319615184),
        T(0.1019301198172404),
        T(0.0832767415767048),
        T(0.0626720483341091),
        T(0.0406014298003869),
        T(0.0176140071391521)
    };

    c10::complex<T> result(T(0), T(0));
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        c10::complex<T> arg_pos = n * theta_pos - z * std::sin(theta_pos);
        c10::complex<T> integrand_pos = theta_pos * c10_complex_math::cos(arg_pos);
        result += weights[j] * integrand_pos;

        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        c10::complex<T> arg_neg = n * theta_neg - z * std::sin(theta_neg);
        c10::complex<T> integrand_neg = theta_neg * c10_complex_math::cos(arg_neg);
        result += weights[j] * integrand_neg;
    }

    return result * half_pi / pi;
}

// Backward pass for Weber function
// Returns (grad_n, grad_z)
//
// Derivative of Weber function w.r.t. z:
// d/dz E_nu(z) = (1/2) * [E_{nu-1}(z) - E_{nu+1}(z)]
//
// Derivative of Weber function w.r.t. nu:
// d/d(nu) E_nu(z) = (1/pi) * integral_0^pi theta * cos(nu*theta - z*sin(theta)) d(theta)
template <typename T>
std::tuple<T, T> weber_e_backward(T grad_output, T n, T z) {
    T grad_n = grad_output * weber_e_derivative_n(n, z);

    T e_nu_minus_1 = weber_e(n - T(1), z);
    T e_nu_plus_1 = weber_e(n + T(1), z);
    T grad_z = grad_output * T(0.5) * (e_nu_minus_1 - e_nu_plus_1);

    return {grad_n, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>> weber_e_backward(
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    c10::complex<T> grad_n = grad_output * weber_e_derivative_n(n, z);

    c10::complex<T> e_nu_minus_1 = weber_e(n - c10::complex<T>(T(1), T(0)), z);
    c10::complex<T> e_nu_plus_1 = weber_e(n + c10::complex<T>(T(1), T(0)), z);
    c10::complex<T> grad_z = grad_output * c10::complex<T>(T(0.5), T(0)) * (e_nu_minus_1 - e_nu_plus_1);

    return {grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
