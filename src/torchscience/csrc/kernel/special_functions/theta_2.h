#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Jacobi theta function theta_2(z, q)
//
// Mathematical definition:
// theta_2(z, q) = 2 * sum_{n=0}^inf q^{(n+1/2)^2} * cos((2n+1)*z)
//
// Domain:
// - z: complex (the argument)
// - q: complex, |q| < 1 (the nome)
//
// Special values:
// - theta_2(0, q) = 2 * sum_{n=0}^inf q^{(n+1/2)^2}
//
// Algorithm:
// Uses the q-series definition with terms summed until convergence.

namespace detail {

template <typename T>
inline T theta2_tolerance() {
    return T(1e-10);
}

template <>
inline float theta2_tolerance<float>() { return 1e-5f; }

template <>
inline double theta2_tolerance<double>() { return 1e-14; }

} // namespace detail

template <typename T>
T theta_2(T z, T q) {
    constexpr int max_terms = 100;
    const T tolerance = detail::theta2_tolerance<T>();

    // Handle q = 0: theta_2(z, 0) = 0
    if (std::abs(q) < tolerance) {
        return T(0);
    }

    T result = T(0);

    for (int n = 0; n < max_terms; ++n) {
        T exp = (n + T(0.5)) * (n + T(0.5));
        T q_power = std::pow(q, exp);

        if (std::abs(q_power) < tolerance) break;

        T term = q_power * std::cos((T(2) * n + T(1)) * z);
        result += term;
    }

    return T(2) * result;
}

template <typename T>
c10::complex<T> theta_2(c10::complex<T> z, c10::complex<T> q) {
    constexpr int max_terms = 100;
    const T tolerance = detail::theta2_tolerance<T>();

    // Handle q = 0: theta_2(z, 0) = 0
    if (std::abs(q) < tolerance) {
        return c10::complex<T>(T(0), T(0));
    }

    c10::complex<T> result(T(0), T(0));

    for (int n = 0; n < max_terms; ++n) {
        T exp = (n + T(0.5)) * (n + T(0.5));
        c10::complex<T> q_power = std::pow(q, c10::complex<T>(exp, T(0)));

        if (std::abs(q_power) < tolerance) break;

        c10::complex<T> arg = c10::complex<T>(T(2) * n + T(1), T(0)) * z;
        c10::complex<T> term = q_power * std::cos(arg);
        result += term;
    }

    return c10::complex<T>(T(2), T(0)) * result;
}

} // namespace torchscience::kernel::special_functions
