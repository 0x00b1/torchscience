#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <vector>

namespace torchscience::kernel::special_functions {

namespace detail {

template <typename T>
constexpr T jacobi_amplitude_am_tolerance();

template <>
constexpr float jacobi_amplitude_am_tolerance<float>() { return 1e-7f; }

template <>
constexpr double jacobi_amplitude_am_tolerance<double>() { return 1e-15; }

template <>
inline c10::Half jacobi_amplitude_am_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 jacobi_amplitude_am_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

template <typename T>
constexpr int jacobi_amplitude_am_max_iter() { return 100; }

} // namespace detail

/**
 * Jacobi amplitude function am(u, m).
 *
 * The Jacobi amplitude am(u, m) is defined as the inverse of the incomplete
 * elliptic integral of the first kind F(phi, m):
 *
 *   am(u, m) = phi  where  u = F(phi, m)
 *
 * The function satisfies:
 *   sn(u, m) = sin(am(u, m))
 *   cn(u, m) = cos(am(u, m))
 *   dn(u, m) = sqrt(1 - m * sin^2(am(u, m)))
 *
 * Implementation uses the Landen descending transformation for efficient
 * and numerically stable computation.
 *
 * @param u The argument (typically the result of an elliptic integral)
 * @param m The parameter (0 <= m <= 1 for standard applications)
 * @return The amplitude phi = am(u, m)
 */
template <typename T>
T jacobi_amplitude_am(T u, T m) {
    const T tolerance = detail::jacobi_amplitude_am_tolerance<T>();
    const int max_iter = detail::jacobi_amplitude_am_max_iter<T>();
    const T pi = T(M_PI);

    // Special case: m = 0
    // am(u, 0) = u (since F(phi, 0) = phi)
    if (m == T(0)) {
        return u;
    }

    // Special case: m = 1
    // am(u, 1) = gd(u) = 2 * atan(exp(u)) - pi/2 (Gudermannian function)
    if (m == T(1)) {
        return T(2) * std::atan(std::exp(u)) - pi / T(2);
    }

    // Handle m slightly outside [0, 1] due to floating point
    if (m < T(0)) {
        // For m < 0, use analytic continuation
        // This is a simplified handling; full complex support is below
        m = T(0);
    }
    if (m > T(1)) {
        m = T(1);
    }

    // Landen descending transformation via AGM (Arithmetic-Geometric Mean)
    // Store both a and c values for back-substitution
    std::vector<T> a_values, c_values;
    a_values.reserve(max_iter + 1);
    c_values.reserve(max_iter + 1);

    T a = T(1);
    T b = std::sqrt(T(1) - m);
    T c = std::sqrt(m);

    // Store initial values (index 0)
    a_values.push_back(a);
    c_values.push_back(c);

    int n = 0;
    while (std::abs(c) > tolerance && n < max_iter) {
        T a_new = (a + b) / T(2);
        T c_new = (a - b) / T(2);
        b = std::sqrt(a * b);
        a = a_new;
        c = c_new;

        // Store values at index n+1
        a_values.push_back(a);
        c_values.push_back(c);
        n++;
    }

    // Compute phi by back-substitution
    // Start with phi_n = 2^n * a_n * u
    T phi = std::ldexp(T(1), n) * a_values[n] * u;

    // Back-substitute: phi_{k-1} = (phi_k + arcsin(c_k * sin(phi_k) / a_k)) / 2
    // for k = n, n-1, ..., 1
    for (int k = n; k >= 1; --k) {
        T sin_phi = std::sin(phi);
        phi = (phi + std::asin(c_values[k] * sin_phi / a_values[k])) / T(2);
    }

    return phi;
}

/**
 * Complex version of Jacobi amplitude function.
 *
 * For complex arguments, we use the same Landen transformation approach
 * but with complex arithmetic.
 */
template <typename T>
c10::complex<T> jacobi_amplitude_am(c10::complex<T> u, c10::complex<T> m) {
    const T tolerance = detail::jacobi_amplitude_am_tolerance<T>();
    const int max_iter = detail::jacobi_amplitude_am_max_iter<T>();
    const c10::complex<T> one(T(1), T(0));
    const c10::complex<T> two(T(2), T(0));
    const T pi = T(M_PI);

    // Special case: m = 0
    if (std::abs(m) < tolerance) {
        return u;
    }

    // Special case: m = 1
    if (std::abs(m - one) < tolerance) {
        return two * std::atan(std::exp(u)) - c10::complex<T>(pi / T(2), T(0));
    }

    // Landen descending transformation via AGM for complex
    // Store both a and c values for back-substitution
    std::vector<c10::complex<T>> a_values, c_values;
    a_values.reserve(max_iter + 1);
    c_values.reserve(max_iter + 1);

    c10::complex<T> a = one;
    c10::complex<T> b = std::sqrt(one - m);
    c10::complex<T> c = std::sqrt(m);

    // Store initial values (index 0)
    a_values.push_back(a);
    c_values.push_back(c);

    int n = 0;
    while (std::abs(c) > tolerance && n < max_iter) {
        c10::complex<T> a_new = (a + b) / two;
        c10::complex<T> c_new = (a - b) / two;
        b = std::sqrt(a * b);
        a = a_new;
        c = c_new;

        // Store values at index n+1
        a_values.push_back(a);
        c_values.push_back(c);
        n++;
    }

    // Compute phi by back-substitution
    // Start with phi_n = 2^n * a_n * u
    c10::complex<T> scale(std::ldexp(T(1), n), T(0));
    c10::complex<T> phi = scale * a_values[n] * u;

    // Back-substitute: phi_{k-1} = (phi_k + arcsin(c_k * sin(phi_k) / a_k)) / 2
    // for k = n, n-1, ..., 1
    for (int k = n; k >= 1; --k) {
        c10::complex<T> sin_phi = std::sin(phi);
        phi = (phi + std::asin(c_values[k] * sin_phi / a_values[k])) / two;
    }

    return phi;
}

} // namespace torchscience::kernel::special_functions
