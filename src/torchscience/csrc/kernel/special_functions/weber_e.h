#pragma once

#include <cmath>
#include <c10/util/complex.h>
#include "anger_j.h"

namespace torchscience::kernel::special_functions {

// Weber function E_nu(z)
// E_nu(z) = (1/pi) * integral_0^pi sin(nu*theta - z*sin(theta)) d(theta)
//
// Using numerical integration (Gauss-Legendre quadrature) which is more stable
// than formulas involving Bessel Y
template <typename T>
T weber_e(T n, T z) {
    const T pi = T(3.14159265358979323846);

    // Use Gaussian quadrature for the integral
    // E_nu(z) = (1/pi) * integral_0^pi sin(nu*theta - z*sin(theta)) d(theta)

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

    // Transform from [-1, 1] to [0, pi]: theta = pi/2 * (t + 1)
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        // Positive node
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        T arg_pos = n * theta_pos - z * std::sin(theta_pos);
        T integrand_pos = std::sin(arg_pos);
        result += weights[j] * integrand_pos;

        // Negative node
        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        T arg_neg = n * theta_neg - z * std::sin(theta_neg);
        T integrand_neg = std::sin(arg_neg);
        result += weights[j] * integrand_neg;
    }

    return result * half_pi / pi;  // = result / 2
}

// Complex version using numerical integration
template <typename T>
c10::complex<T> weber_e(c10::complex<T> n, c10::complex<T> z) {
    const T pi = T(3.14159265358979323846);

    // Use Gaussian quadrature for the integral
    // E_nu(z) = (1/pi) * integral_0^pi sin(nu*theta - z*sin(theta)) d(theta)

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

    c10::complex<T> result(T(0), T(0));

    // Transform from [-1, 1] to [0, pi]: theta = pi/2 * (t + 1)
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        // Positive node
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        c10::complex<T> arg_pos = n * theta_pos - z * std::sin(theta_pos);
        c10::complex<T> integrand_pos = c10_complex_math::sin(arg_pos);
        result += weights[j] * integrand_pos;

        // Negative node
        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        c10::complex<T> arg_neg = n * theta_neg - z * std::sin(theta_neg);
        c10::complex<T> integrand_neg = c10_complex_math::sin(arg_neg);
        result += weights[j] * integrand_neg;
    }

    return result * half_pi / pi;  // = result / 2
}

} // namespace torchscience::kernel::special_functions
