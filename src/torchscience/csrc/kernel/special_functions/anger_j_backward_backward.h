#pragma once

#include <tuple>
#include <cmath>
#include <c10/util/complex.h>

#include "anger_j.h"
#include "anger_j_backward.h"

namespace torchscience::kernel::special_functions {

// Helper: compute d²/dn² J_nu(z) = -(1/pi) * integral_0^pi theta^2 * cos(nu*theta - z*sin(theta)) d(theta)
template <typename T>
T anger_j_second_derivative_n2(T n, T z) {
    const T pi = T(3.14159265358979323846);

    const T nodes[10] = {
        T(0.0765265211334973), T(0.2277858511416451), T(0.3737060887154195),
        T(0.5108670019508271), T(0.6360536807265150), T(0.7463319064601508),
        T(0.8391169718222188), T(0.9122344282513259), T(0.9639719272779138),
        T(0.9931285991850949)
    };
    const T weights[10] = {
        T(0.1527533871307258), T(0.1491729864726037), T(0.1420961093183820),
        T(0.1316886384491766), T(0.1181945319615184), T(0.1019301198172404),
        T(0.0832767415767048), T(0.0626720483341091), T(0.0406014298003869),
        T(0.0176140071391521)
    };

    T result = T(0);
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        T arg_pos = n * theta_pos - z * std::sin(theta_pos);
        T integrand_pos = -theta_pos * theta_pos * std::cos(arg_pos);
        result += weights[j] * integrand_pos;

        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        T arg_neg = n * theta_neg - z * std::sin(theta_neg);
        T integrand_neg = -theta_neg * theta_neg * std::cos(arg_neg);
        result += weights[j] * integrand_neg;
    }
    return result * half_pi / pi;
}

// Helper: compute d²/dndz J_nu(z) = (1/pi) * integral_0^pi theta * sin(theta) * cos(nu*theta - z*sin(theta)) d(theta)
template <typename T>
T anger_j_second_derivative_nz(T n, T z) {
    const T pi = T(3.14159265358979323846);

    const T nodes[10] = {
        T(0.0765265211334973), T(0.2277858511416451), T(0.3737060887154195),
        T(0.5108670019508271), T(0.6360536807265150), T(0.7463319064601508),
        T(0.8391169718222188), T(0.9122344282513259), T(0.9639719272779138),
        T(0.9931285991850949)
    };
    const T weights[10] = {
        T(0.1527533871307258), T(0.1491729864726037), T(0.1420961093183820),
        T(0.1316886384491766), T(0.1181945319615184), T(0.1019301198172404),
        T(0.0832767415767048), T(0.0626720483341091), T(0.0406014298003869),
        T(0.0176140071391521)
    };

    T result = T(0);
    T half_pi = pi / T(2);

    for (int j = 0; j < 10; ++j) {
        T t_pos = nodes[j];
        T theta_pos = half_pi * (t_pos + T(1));
        T sin_theta_pos = std::sin(theta_pos);
        T arg_pos = n * theta_pos - z * sin_theta_pos;
        T integrand_pos = theta_pos * sin_theta_pos * std::cos(arg_pos);
        result += weights[j] * integrand_pos;

        T t_neg = -nodes[j];
        T theta_neg = half_pi * (t_neg + T(1));
        T sin_theta_neg = std::sin(theta_neg);
        T arg_neg = n * theta_neg - z * sin_theta_neg;
        T integrand_neg = theta_neg * sin_theta_neg * std::cos(arg_neg);
        result += weights[j] * integrand_neg;
    }
    return result * half_pi / pi;
}

// Second-order backward pass for Anger function
// Returns (grad_grad_output, grad_n, grad_z)
template <typename T>
std::tuple<T, T, T> anger_j_backward_backward(
    T gg_n,
    T gg_z,
    T grad_output,
    T n,
    T z
) {
    // First derivatives
    T dfdn = anger_j_derivative_n(n, z);
    T j_nu_minus_1 = anger_j(n - T(1), z);
    T j_nu_plus_1 = anger_j(n + T(1), z);
    T dfdz = T(0.5) * (j_nu_minus_1 - j_nu_plus_1);

    // Second derivatives
    T d2f_dn2 = anger_j_second_derivative_n2(n, z);
    T d2f_dndz = anger_j_second_derivative_nz(n, z);

    T j_nu = anger_j(n, z);
    T j_nu_minus_2 = anger_j(n - T(2), z);
    T j_nu_plus_2 = anger_j(n + T(2), z);
    T d2f_dz2 = T(0.25) * (j_nu_minus_2 - T(2) * j_nu + j_nu_plus_2);

    // Compute outputs
    T grad_grad_output = gg_n * dfdn + gg_z * dfdz;
    T grad_n = grad_output * (gg_n * d2f_dn2 + gg_z * d2f_dndz);
    T grad_z = grad_output * (gg_n * d2f_dndz + gg_z * d2f_dz2);

    return {grad_grad_output, grad_n, grad_z};
}

// Complex version
template <typename T>
std::tuple<c10::complex<T>, c10::complex<T>, c10::complex<T>> anger_j_backward_backward(
    c10::complex<T> gg_n,
    c10::complex<T> gg_z,
    c10::complex<T> grad_output,
    c10::complex<T> n,
    c10::complex<T> z
) {
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const c10::complex<T> half(T(0.5), T(0));
    const c10::complex<T> quarter(T(0.25), T(0));

    // First derivatives (using integral formulas)
    c10::complex<T> dfdn = anger_j_derivative_n(n, z);
    c10::complex<T> j_nu_minus_1 = anger_j(n - one, z);
    c10::complex<T> j_nu_plus_1 = anger_j(n + one, z);
    c10::complex<T> dfdz = half * (j_nu_minus_1 - j_nu_plus_1);

    // Second derivatives - for simplicity, use numerical approach with recurrence
    c10::complex<T> j_nu = anger_j(n, z);
    c10::complex<T> j_nu_minus_2 = anger_j(n - two, z);
    c10::complex<T> j_nu_plus_2 = anger_j(n + two, z);
    c10::complex<T> d2f_dz2 = quarter * (j_nu_minus_2 - two * j_nu + j_nu_plus_2);

    // For mixed derivatives, use finite differences on the derivative
    c10::complex<T> eps(T(1e-6), T(0));
    c10::complex<T> dfdn_plus = anger_j_derivative_n(n + eps, z);
    c10::complex<T> dfdn_minus = anger_j_derivative_n(n - eps, z);
    c10::complex<T> d2f_dn2 = (dfdn_plus - dfdn_minus) / (two * eps);
    c10::complex<T> d2f_dndz = (anger_j_derivative_n(n, z + eps) - anger_j_derivative_n(n, z - eps)) / (two * eps);

    // Compute outputs
    c10::complex<T> grad_grad_output = gg_n * std::conj(dfdn) + gg_z * std::conj(dfdz);
    c10::complex<T> grad_n = grad_output * (gg_n * std::conj(d2f_dn2) + gg_z * std::conj(d2f_dndz));
    c10::complex<T> grad_z = grad_output * (gg_n * std::conj(d2f_dndz) + gg_z * std::conj(d2f_dz2));

    return {grad_grad_output, grad_n, grad_z};
}

} // namespace torchscience::kernel::special_functions
