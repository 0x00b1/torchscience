#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Jacobi theta function theta_4(z, q)
//
// Mathematical definition:
// theta_4(z, q) = 1 + 2 * sum_{n=1}^inf (-1)^n * q^{n^2} * cos(2*n*z)
//
// Domain:
// - z: complex (the argument)
// - q: complex, |q| < 1 (the nome)
//
// Special values:
// - theta_4(0, q) = 1 + 2 * sum_{n=1}^inf (-1)^n * q^{n^2}
// - theta_4(z, 0) = 1
//
// Algorithm:
// Uses the q-series definition with terms summed until convergence.

namespace detail {

template <typename T>
inline T theta4_tolerance() {
    return T(1e-10);
}

template <>
inline float theta4_tolerance<float>() { return 1e-5f; }

template <>
inline double theta4_tolerance<double>() { return 1e-14; }

} // namespace detail

template <typename T>
T theta_4(T z, T q) {
    constexpr int max_terms = 100;
    const T tolerance = detail::theta4_tolerance<T>();

    // Handle q = 0: theta_4(z, 0) = 1
    if (std::abs(q) < tolerance) {
        return T(1);
    }

    T result = T(1);  // Start with the n=0 term
    T sign = T(-1);   // First term (n=1) has sign (-1)^1 = -1

    for (int n = 1; n <= max_terms; ++n) {
        T exp = T(n) * T(n);
        T q_power = std::pow(q, exp);

        if (std::abs(q_power) < tolerance) break;

        T term = sign * q_power * std::cos(T(2) * T(n) * z);
        result += T(2) * term;
        sign = -sign;
    }

    return result;
}

template <typename T>
c10::complex<T> theta_4(c10::complex<T> z, c10::complex<T> q) {
    constexpr int max_terms = 100;
    const T tolerance = detail::theta4_tolerance<T>();

    c10::complex<T> one(T(1), T(0));
    c10::complex<T> neg_one(T(-1), T(0));

    // Handle q = 0: theta_4(z, 0) = 1
    if (std::abs(q) < tolerance) {
        return one;
    }

    c10::complex<T> result = one;  // Start with the n=0 term
    c10::complex<T> sign = neg_one;  // First term (n=1) has sign -1

    for (int n = 1; n <= max_terms; ++n) {
        T exp = T(n) * T(n);
        c10::complex<T> q_power = std::pow(q, c10::complex<T>(exp, T(0)));

        if (std::abs(q_power) < tolerance) break;

        c10::complex<T> arg = c10::complex<T>(T(2) * T(n), T(0)) * z;
        c10::complex<T> term = sign * q_power * std::cos(arg);
        result += c10::complex<T>(T(2), T(0)) * term;
        sign = sign * neg_one;
    }

    return result;
}

} // namespace torchscience::kernel::special_functions
