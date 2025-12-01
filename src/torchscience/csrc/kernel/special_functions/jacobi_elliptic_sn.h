#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/util/complex.h>
#include <cmath>
#include <limits>

namespace torchscience::kernel::special_functions {

// Jacobi elliptic function sn(u, m)
//
// Mathematical definition:
// sn(u, m) = sin(am(u, m))
//
// where am(u, m) is the Jacobi amplitude function defined implicitly by:
// u = integral from 0 to am(u, m) of d(theta) / sqrt(1 - m * sin^2(theta))
//
// Domain:
// - u: real or complex
// - m: elliptic parameter, conventionally 0 <= m <= 1 for real values
//   - m = 0: sn(u, 0) = sin(u)
//   - m = 1: sn(u, 1) = tanh(u)
//   - For m < 0 or m > 1, use analytic continuation
//
// Special values:
// - sn(0, m) = 0 for all m
// - sn(K(m), m) = 1 where K(m) is the complete elliptic integral of the first kind
// - sn(-u, m) = -sn(u, m) (odd function in u)
//
// Algorithm:
// Uses the Landen transformation (descending Landen sequence) to compute
// the Jacobi amplitude am(u, m), then returns sin(am).
//
// The Landen transformation successively reduces m towards 0 by:
// m_{n+1} = ((1 - sqrt(1 - m_n)) / (1 + sqrt(1 - m_n)))^2
//
// For small m, sn(u, m) ≈ sin(u)
// For m = 1, sn(u, 1) = tanh(u)

namespace detail {

template <typename T>
constexpr int jacobi_max_iterations() { return 50; }

template <typename T>
constexpr T jacobi_tolerance();

template <>
constexpr float jacobi_tolerance<float>() { return 1e-7f; }

template <>
constexpr double jacobi_tolerance<double>() { return 1e-15; }

template <>
inline c10::Half jacobi_tolerance<c10::Half>() { return c10::Half(1e-3f); }

template <>
inline c10::BFloat16 jacobi_tolerance<c10::BFloat16>() { return c10::BFloat16(1e-3f); }

} // namespace detail

template <typename T>
T jacobi_elliptic_sn(T u, T m) {
    // Handle special case m = 0: sn(u, 0) = sin(u)
    if (m == T(0)) {
        return std::sin(u);
    }

    // Handle special case m = 1: sn(u, 1) = tanh(u)
    if (m == T(1)) {
        return std::tanh(u);
    }

    // Handle u = 0: sn(0, m) = 0
    if (u == T(0)) {
        return T(0);
    }

    const T tol = detail::jacobi_tolerance<T>();
    const int max_iter = detail::jacobi_max_iterations<T>();

    // For m close to 1, use the hyperbolic approximation
    if (m > T(1) - tol) {
        // Near m = 1: sn(u, m) ≈ tanh(u) - (1-m)/4 * (sinh(u)*cosh(u) - u) * sech^2(u)
        return std::tanh(u);
    }

    // For m close to 0, use the circular approximation
    if (std::abs(m) < tol) {
        // Near m = 0: sn(u, m) ≈ sin(u) - m/4 * (u - sin(u)*cos(u)) * cos(u)
        return std::sin(u);
    }

    // Handle negative m using the transformation:
    // sn(u, m) = (1/sqrt(1-m)) * sn(u*sqrt(1-m), m/(m-1)) for m < 0
    // This is the reciprocal modulus transformation
    if (m < T(0)) {
        T m1 = T(1) - m;  // m1 > 1
        T k1 = std::sqrt(m1);
        T m_new = m / (m - T(1));  // 0 < m_new < 1
        return jacobi_elliptic_sn(u * k1, m_new) / k1;
    }

    // Handle m > 1 using the reciprocal modulus transformation:
    // sn(u, m) = (1/sqrt(m)) * sn(u*sqrt(m), 1/m)
    if (m > T(1)) {
        T sqrt_m = std::sqrt(m);
        return jacobi_elliptic_sn(u * sqrt_m, T(1) / m) / sqrt_m;
    }

    // Main algorithm: Landen transformation (descending sequence)
    // Store the sequence of moduli for backward reconstruction
    T a[64];  // Arithmetic-geometric mean sequence
    T g[64];  // Geometric mean sequence
    T c[64];  // c_n = (a_n - g_n) / 2

    T a_n = T(1);
    T g_n = std::sqrt(T(1) - m);
    T c_n = std::sqrt(m);

    int n = 0;
    a[0] = a_n;
    g[0] = g_n;
    c[0] = c_n;

    // Compute the descending Landen transformation until c_n is negligible
    while (std::abs(c_n) > tol && n < max_iter) {
        T a_next = (a_n + g_n) / T(2);
        T g_next = std::sqrt(a_n * g_n);
        c_n = (a_n - g_n) / T(2);

        a_n = a_next;
        g_n = g_next;
        n++;

        a[n] = a_n;
        g[n] = g_n;
        c[n] = c_n;
    }

    // Compute phi_n = 2^n * a_n * u
    T phi = std::ldexp(a_n * u, n);

    // Backward recurrence to find amplitude
    // phi_{n-1} = (phi_n + arcsin(c_n/a_n * sin(phi_n))) / 2
    for (int i = n; i > 0; i--) {
        T sin_phi = std::sin(phi);
        phi = (phi + std::asin(c[i] / a[i] * sin_phi)) / T(2);
    }

    // sn(u, m) = sin(am(u, m)) = sin(phi)
    return std::sin(phi);
}

template <typename T>
c10::complex<T> jacobi_elliptic_sn(c10::complex<T> u, c10::complex<T> m) {
    // Handle special case m = 0: sn(u, 0) = sin(u)
    if (std::abs(m) < detail::jacobi_tolerance<T>()) {
        return std::sin(u);
    }

    // Handle special case m = 1: sn(u, 1) = tanh(u)
    c10::complex<T> one(T(1), T(0));
    if (std::abs(m - one) < detail::jacobi_tolerance<T>()) {
        return std::tanh(u);
    }

    // Handle u = 0: sn(0, m) = 0
    if (std::abs(u) < detail::jacobi_tolerance<T>()) {
        return c10::complex<T>(T(0), T(0));
    }

    const T tol = detail::jacobi_tolerance<T>();
    const int max_iter = detail::jacobi_max_iterations<T>();

    // Complex Landen transformation
    c10::complex<T> a[64];
    c10::complex<T> g[64];
    c10::complex<T> c[64];

    c10::complex<T> a_n = one;
    c10::complex<T> g_n = std::sqrt(one - m);
    c10::complex<T> c_n = std::sqrt(m);

    int n = 0;
    a[0] = a_n;
    g[0] = g_n;
    c[0] = c_n;

    c10::complex<T> two(T(2), T(0));

    while (std::abs(c_n) > tol && n < max_iter) {
        c10::complex<T> a_next = (a_n + g_n) / two;
        c10::complex<T> g_next = std::sqrt(a_n * g_n);
        c_n = (a_n - g_n) / two;

        a_n = a_next;
        g_n = g_next;
        n++;

        a[n] = a_n;
        g[n] = g_n;
        c[n] = c_n;
    }

    // Compute phi_n = 2^n * a_n * u
    c10::complex<T> phi = a_n * u * T(std::ldexp(1.0, n));

    // Backward recurrence
    for (int i = n; i > 0; i--) {
        c10::complex<T> sin_phi = std::sin(phi);
        c10::complex<T> arg = c[i] / a[i] * sin_phi;
        // For complex asin, we use the identity: asin(z) = -i * log(iz + sqrt(1 - z^2))
        phi = (phi + std::asin(arg)) / two;
    }

    return std::sin(phi);
}

} // namespace torchscience::kernel::special_functions
